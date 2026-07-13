"""MemVR core (Zou et al., ICML 2025): per-generation ``MemVRBuffer`` state, the
entropy trigger (logit-lens over the last position), the bilinear FFN adapter
injection, and the hook/mlp-wrapper installer. Orthogonal to SPARC; coexists via
its own forward hooks and a ``mlp.forward`` wrapper, never touching ``attn.py``.
"""

from __future__ import annotations

import math
from types import MethodType

import torch

try:
    from vr_modality_bias.utils.attn import decoder_of
except ModuleNotFoundError:
    import sys

    from pyprojroot import here

    sys.path.insert(0, str(here()))
    from src.vr_modality_bias.utils.attn import decoder_of

__all__ = [
    "MemVRBuffer",
    "install_memvr_hooks",
    "memvr_adapter_mix",
    "normalized_topk_entropy",
]

_EPS = 1e-12


def _mlp_module_of(layer) -> object:
    """Return the layer's feed-forward submodule.

    Mirrors ``utils.attn._attention_module_of``: families name the block
    differently (``mlp`` for Llama / Idefics3 / SmolVLM / Qwen; ``feed_forward``
    for some ports). The injection wrapper needs it plus its ``up_proj`` /
    ``down_proj`` for the magnitude anchors.
    """
    for cand in ("mlp", "feed_forward"):
        if hasattr(layer, cand):
            return getattr(layer, cand)
    raise AttributeError(
        f"decoder layer {type(layer).__name__} has neither `.mlp` nor "
        f"`.feed_forward` -- extend `_mlp_module_of` in utils/memvr.py."
    )


def normalized_topk_entropy(logits: torch.Tensor, top_k: int = 10) -> torch.Tensor:
    """Entropy of the softmax over the top-``k`` logits, normalized to ``[0, 1]``.

    The MemVR trigger's uncertainty measure: take the ``k`` largest logits,
    softmax WITHIN that set, and normalize the Shannon entropy by ``log(k)`` so a
    uniform top-k is 1.0 and an all-in-one top-k is ~0.0. Computed in float32.

    Clamped to ``[0, 1]``: entropy over ``k`` outcomes never exceeds ``log(k)``
    mathematically, and the clamp removes the float residue so ``gamma = 1.0`` is
    an exact "never fires" gate (the parametric neutrality gate).
    """
    logits = logits.float()
    k = min(int(top_k), logits.shape[-1])
    if k <= 1:
        return torch.zeros(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    topv = torch.topk(logits, k, dim=-1).values
    p = torch.softmax(topv, dim=-1)
    entropy = -(p * torch.log(p.clamp_min(_EPS))).sum(dim=-1)
    return (entropy / math.log(k)).clamp(min=0.0, max=1.0)


def memvr_adapter_mix(
    x: torch.Tensor,
    ffn_out: torch.Tensor,
    Z: torch.Tensor | None,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend the pure FFN output with the visual-retracing adapter.

    Returns ``(1 - alpha) * FFN(x) + alpha * adapter_norm(x)`` where
    ``adapter(x) = (x @ Z1^T) @ Z2`` with ``Z1 = s1 * Z`` and ``Z2 = s2 * Z``
    (Z = the projected visual tokens, ``[N_img, d]``). The three magnitude
    normalizations from the official MemVR code:

    * ``s1 = mean|up_proj.weight| / mean|Z|`` scales the input projection,
    * ``s2 = mean|down_proj.weight| / mean|Z|`` scales the output projection,
    * the adapter output is rescaled to ``mean|FFN(x)|``.

    Factors are computed in float32; the result is cast back to ``ffn_out``'s
    dtype. NaN guard: ``alpha == 0``, ``Z is None``, ``mean|Z| == 0``, or an
    all-zero adapter (``mean|adapter| == 0``) short-circuit to the pure FFN
    without any division. ``alpha == 0`` returning ``ffn_out`` unchanged is the
    parametric-neutrality gate, and it also matches the official code's own NaN
    hazard (dividing by ``mean|adapter| == 0``).
    """
    alpha = float(alpha)
    if alpha == 0.0 or Z is None:
        return ffn_out
    Zf = Z.float()
    mean_z = Zf.abs().mean()
    if float(mean_z) == 0.0:
        return ffn_out
    s1 = up_weight.detach().float().abs().mean() / mean_z
    s2 = down_weight.detach().float().abs().mean() / mean_z
    z1 = s1 * Zf
    z2 = s2 * Zf
    xf = x.float()
    adapter = torch.matmul(torch.matmul(xf, z1.transpose(-1, -2)), z2)
    mean_adapter = adapter.abs().mean()
    if float(mean_adapter) == 0.0:
        return ffn_out
    ffn_f = ffn_out.float()
    mean_ffn = ffn_f.abs().mean()
    adapter_norm = adapter * (mean_ffn / mean_adapter)
    mixed = (1.0 - alpha) * ffn_f + alpha * adapter_norm
    return mixed.to(ffn_out.dtype)


class MemVRBuffer:
    """Per-generation MemVR state and instrumentation.

    Configuration (``gamma``, ``alpha``, ``window``, ``top_k``) is set by
    ``enable_memvr`` from the hyperparameters; the hooks read it at call time.
    ``image_positions`` must be set by the caller before each prefill (as with
    SPARC's buffer), because ``Z`` is read from the input hidden states at those
    positions.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        # Projected visual tokens captured at the prefill (float32, [N_img, d]).
        self.Z = None
        # Global prompt positions where input_ids == image_token_id.
        self.image_positions = None
        # Effective (inclusive) layer window in which the trigger may fire.
        self.window = (0, 0)
        self.gamma = 1.0
        self.alpha = 0.0
        self.top_k = 10
        # Layer whose mlp must inject this forward (l+1 of the firing layer), or
        # None. Reset at the start of every decoder forward so a temporary
        # injection never leaks across forwards.
        self.armed_layer = None
        self.fired_this_forward = False
        self.current_is_prefill = False
        # ---- instrumentation ----
        self.n_fires_total = 0
        self.fired_in_prefill = False
        self.fire_layer = None
        self.fire_entropy = None

    def update_image_positions(self, positions: torch.Tensor) -> None:
        """Store the per-image global positions used to read ``Z`` at prefill."""
        self.image_positions = positions

    def begin_forward(self, is_prefill: bool) -> None:
        """Reset the per-forward firing state at the start of a decoder forward.

        Enforces at-most-one injection per forward (``fired_this_forward``) and
        drops any stale arming so a temporary injection never persists.
        """
        self.fired_this_forward = False
        self.armed_layer = None
        self.current_is_prefill = bool(is_prefill)

    def capture_Z(self, hidden: torch.Tensor) -> None:
        """Capture the projected visual tokens from the prefill input hiddens.

        ``hidden`` is the input to decoder layer 0 (post-projector, pre-decoder),
        ``[1, seq, d]``; ``Z`` is the ``image_positions`` rows, upcast to float32.
        """
        if self.image_positions is None:
            raise RuntimeError(
                "MemVR needs update_image_positions() before the prefill: Z is "
                "read from the input hidden states at the image positions and "
                "cannot be located without them."
            )
        pos = self.image_positions.to(hidden.device)
        self.Z = hidden[0].index_select(0, pos).detach().float()

    def arm(self, target_layer: int, entropy: float, fire_layer: int) -> None:
        """Arm the injection for ``target_layer`` and record the trigger."""
        self.armed_layer = int(target_layer)
        self.fired_this_forward = True
        self.n_fires_total += 1
        self.fire_layer = int(fire_layer)
        self.fire_entropy = float(entropy)
        if self.current_is_prefill:
            self.fired_in_prefill = True


def _logit_lens_entropy(hidden_last, final_norm, lm_head, top_k) -> torch.Tensor:
    """Normalized top-k entropy of the last position through the logit lens.

    Entropy is computed over the RAW logits: no ``logits_processor`` is applied,
    a conscious divergence from the official MemVR code (which runs the decoding
    repetition penalty first). Our POPE uses rp=1.0 anyway; keeping it raw drops
    a dependency on the generation setup.
    """
    with torch.no_grad():
        logits = lm_head(final_norm(hidden_last))
    return normalized_topk_entropy(logits, top_k)


def _make_memvr_mlp_forward(original_forward, buffer: MemVRBuffer, layer_idx: int):
    """Wrap ``mlp.forward`` so the armed layer blends in the adapter, then disarms.

    Installed on every layer and gated by ``armed_layer`` (the SPARC
    "patch-all, act-conditionally" pattern) rather than installed dynamically on
    the firing layer: install/remove stays symmetric and there is no re-entrant
    hook mutation mid-forward. Disarms before computing so a single injection
    happens per forward even if the NaN guard short-circuits.
    """

    def _memvr_mlp_forward(self, *args, **kwargs):
        ffn_out = original_forward(*args, **kwargs)
        if buffer.armed_layer != layer_idx:
            return ffn_out
        buffer.armed_layer = None
        x = args[0] if args else kwargs.get("hidden_states")
        up = getattr(self, "up_proj", None)
        down = getattr(self, "down_proj", None)
        if up is None or down is None:
            raise AttributeError(
                f"MemVR injection needs `up_proj`/`down_proj` on "
                f"{type(self).__name__} for the magnitude anchors; not found."
            )
        return memvr_adapter_mix(x, ffn_out, buffer.Z, up.weight, down.weight, buffer.alpha)

    return _memvr_mlp_forward


class MemVRInstallation:
    """Handle returned by ``install_memvr_hooks``; ``remove()`` undoes everything."""

    def __init__(self, hook_handles, original_mlp_forwards):
        self._hook_handles = hook_handles
        self._original_mlp_forwards = original_mlp_forwards

    def remove(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        for mlp, original in self._original_mlp_forwards:
            mlp.forward = original
        self._hook_handles = []
        self._original_mlp_forwards = []


def install_memvr_hooks(model_wrapper, buffer: MemVRBuffer) -> MemVRInstallation:
    """Install the MemVR machinery on a loaded model; return a removable handle.

    Three parts, all reading ``buffer`` at call time:

    * a layer-0 forward pre-hook that begins each forward (resets the
      per-forward firing state, detects prefill by ``seq_len > 1``) and captures
      ``Z`` on the prefill;
    * a per-layer forward hook computing the last-position entropy via the logit
      lens and arming layer ``l+1`` when it exceeds ``gamma`` inside the window,
      at most once per forward;
    * a per-layer ``mlp.forward`` wrapper that injects on the armed layer.
    """
    model = model_wrapper._model
    decoder = decoder_of(model)
    layers = decoder.layers
    final_norm = model_wrapper.get_final_norm()
    lm_head = model_wrapper.get_lm_head()

    hook_handles = []
    original_mlp_forwards = []

    def _layer0_pre_hook(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        # bsz == 1 throughout the codebase; a prefill has seq_len > 1, a decode
        # step exactly 1. This is the clean prefill signal available to a hook
        # (the cache is not reliably reachable from the hook args across versions).
        is_prefill = hidden.shape[1] > 1
        buffer.begin_forward(is_prefill)
        if is_prefill:
            buffer.capture_Z(hidden)

    hook_handles.append(
        layers[0].register_forward_pre_hook(_layer0_pre_hook, with_kwargs=True)
    )

    for i, layer in enumerate(layers):

        def _entropy_hook(module, inp, output, layer_idx=i):
            if buffer.fired_this_forward:
                return
            start, end = buffer.window
            if not (start <= layer_idx <= end):
                return
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            hidden_last = hidden[:, -1, :]
            entropy = _logit_lens_entropy(hidden_last, final_norm, lm_head, buffer.top_k)
            entropy_val = float(entropy.reshape(-1)[0])
            if entropy_val > buffer.gamma:
                buffer.arm(layer_idx + 1, entropy_val, layer_idx)

        hook_handles.append(layer.register_forward_hook(_entropy_hook))

        mlp = _mlp_module_of(layer)
        original_forward = mlp.forward
        original_mlp_forwards.append((mlp, original_forward))
        mlp.forward = MethodType(
            _make_memvr_mlp_forward(original_forward, buffer, i), mlp
        )

    return MemVRInstallation(hook_handles, original_mlp_forwards)
