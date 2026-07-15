"""VISTA core: per-question ``VistaBuffer`` state, the
visual steering vector (VSV) extraction from two clean prefills, the norm-
preserving MLP steering with the adaptive anti-visual weight, the SLA logit
mix, and the mlp-wrapper installer. Orthogonal to SPARC; shares the
``mlp.forward`` slot with MemVR but the two never steer at once (guarded).
"""

from __future__ import annotations

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
    "VistaBuffer",
    "build_negative_inputs",
    "compute_vsv",
    "install_vista_hooks",
    "sla_mix_logits",
    "vista_adaptive_factor",
    "vista_steer",
]

_EPS = 1e-12


def _mlp_module_of(layer) -> object:
    """Return the layer's feed-forward submodule (mirrors ``memvr._mlp_module_of``).

    Kept local so VISTA does not depend on a private name in ``utils.memvr``.
    """
    for cand in ("mlp", "feed_forward"):
        if hasattr(layer, cand):
            return getattr(layer, cand)
    raise AttributeError(
        f"decoder layer {type(layer).__name__} has neither `.mlp` nor "
        f"`.feed_forward` -- extend `_mlp_module_of` in utils/vista.py."
    )


# ------------------------------------------------------------------ steering


def vista_adaptive_factor(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The per-position anti-visual weight ``1 + max(0, cos(x, -v))``.

    ``cos(x, -v) = -(x_hat . v_hat)``: it is positive exactly when the state ``x``
    points AGAINST the visual direction ``v``, so the factor rises to 2 when the
    state has fully drifted anti-visual and stays 1 when it is aligned. Computed
    in float32; returns a ``[.., 1]`` tensor (keepdim) for broadcasting.
    """
    xf = x.float()
    vf = v.float()
    x_unit = xf / xf.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    v_unit = vf / vf.norm().clamp_min(_EPS)
    cos_neg = -(x_unit * v_unit).sum(dim=-1, keepdim=True)
    return 1.0 + cos_neg.clamp_min(0.0)


def vista_steer(x: torch.Tensor, v: torch.Tensor | None, lam: float) -> torch.Tensor:
    """Norm-preserving visual steering of an MLP output ``x`` by the VSV ``v``.

    ``y = lam * (1 + max(0, cos(x, -v))) * v_hat``;
    ``x_new = normalize(normalize(x) + y) * ||x||``. The renormalization keeps
    ``||x_new|| == ||x||`` (the steering rotates the state, it does not inflate
    it). Computed in float32, returned in ``x``'s dtype.

    Short-circuit (the parametric gate): ``lam == 0`` or ``v is None`` returns
    ``x`` unchanged with NO arithmetic (no normalize round-trip, no casts). This
    diverges from the official code deliberately (there ``lam=0`` is not
    bit-identical); see planning section 1.6. The window / installed short-
    circuit lives in the wrapper.
    """
    if float(lam) == 0.0 or v is None:
        return x
    xf = x.float()
    vf = v.float()
    x_norm = xf.norm(dim=-1, keepdim=True)
    x_unit = xf / x_norm.clamp_min(_EPS)
    v_unit = vf / vf.norm().clamp_min(_EPS)
    lambda_sim = vista_adaptive_factor(x, v)
    y = float(lam) * lambda_sim * v_unit
    steered = x_unit + y
    steered_unit = steered / steered.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return (steered_unit * x_norm).to(x.dtype)


def sla_mix_logits(hidden_states, lm_head, logits, alpha: float, window) -> torch.Tensor:
    """SLA: blend the final logits with the mean of window-layer logit lenses.

    ``alpha * mean(lm_head(h_l), l in window) + (1 - alpha) * logits``. The
    per-layer projection uses ``lm_head`` WITHOUT the final norm, faithful to the
    official VISTA and to the convention of ``metrics/kl.py`` (which also skips
    the norm). ``hidden_states`` is the per-layer sequence (e.g.
    ``outputs.hidden_states[1:]``); ``window`` is an inclusive ``(start, end)``
    into it. Computed in float32, returned in ``logits``'s dtype.

    ``alpha == 0`` returns ``logits`` unchanged; ``alpha == 1`` returns the window
    mean. Pure and stateless; used by the diagnostic in Etapa 2.
    """
    alpha = float(alpha)
    if alpha == 0.0:
        return logits
    start, end = int(window[0]), int(window[1])
    projected = [lm_head(hidden_states[layer]).float() for layer in range(start, end + 1)]
    mean_logits = torch.stack(projected, dim=0).mean(dim=0)
    logits_f = logits.float()
    mixed = alpha * mean_logits + (1.0 - alpha) * logits_f
    return mixed.to(logits.dtype)


# ------------------------------------------------------------------ VSV extraction


def _strip_image_items(messages):
    """Drop every image content item from chat messages (keep everything else)."""
    stripped = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            kept = [
                item
                for item in content
                if not (isinstance(item, dict) and item.get("type") in ("image", "image_url"))
            ]
            stripped.append({**msg, "content": kept})
        else:
            stripped.append(msg)
    return stripped


def build_negative_inputs(model_wrapper, prompt: str):
    """Build the text-only (image-free) prefill inputs for the VSV negative.

    The same chat messages the positive prefill uses, with the image item
    stripped, run through ``apply_chat_template`` + the processor WITHOUT any
    image. Asserts programmatically that no token equals the model's
    ``image_token_id`` (a malformed template that still emits image placeholders
    would poison the negative), matching planning gate 3.
    """
    processor = model_wrapper._processor  # noqa: SLF001
    messages = _strip_image_items(model_wrapper._build_messages(prompt, None))  # noqa: SLF001
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=[prefix_text], return_tensors="pt")
    image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
    if bool((inputs["input_ids"] == image_token_id).any()):
        raise AssertionError(
            "The negative (text-only) input still contains image tokens; the "
            "image item was not stripped from the messages. The VSV negative "
            "must be pure text."
        )
    return inputs


def _move_inputs(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def _last_hidden_per_layer(model, inputs, device) -> torch.Tensor:
    """Stack the last-position hidden of each decoder layer (embedding excluded)."""
    inputs = _move_inputs(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)
    layers = outputs.hidden_states[1:]  # [1:] drops the embedding output
    return torch.stack([h[0, -1, :].detach().float() for h in layers], dim=0)


def compute_vsv(model_wrapper, pos_inputs, neg_inputs) -> torch.Tensor:
    """The visual steering vector: ``[L, d]`` float32, positive minus negative.

    Runs the two prefills (image and text-only) with ``output_hidden_states``,
    reads each layer's LAST position (the two sequences have different lengths,
    so the last position of each is the right anchor), and returns
    ``pos - neg``. A single pos/neg pair makes the official PCA rank-1 step
    degenerate into this direct difference (recorded equivalence).

    NOTE: this MUST run on a clean model, outside any intervention context; the
    orchestration guarantees that (compute the VSV before opening enable_sparc /
    enable_vista), as the official code computes the vector before add_vsv_layers.
    """
    model = model_wrapper._model  # noqa: SLF001
    device = next(model.parameters()).device
    pos_last = _last_hidden_per_layer(model, pos_inputs, device)
    neg_last = _last_hidden_per_layer(model, neg_inputs, device)
    return pos_last - neg_last


# ------------------------------------------------------------------ buffer


class VistaBuffer:
    """Per-question VISTA state and instrumentation.

    ``lam`` and ``window`` are set by ``enable_vista``; the VSV enters explicitly
    via :meth:`set_vsv` after the (clean) extraction, so the wrapper stays inert
    until the caller arms it.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        # Per-layer steering vector, float32 [L, d], or None (inert).
        self.vsv = None
        self.lam = 0.0
        # Inclusive (start, end) layer window, or None meaning all layers.
        self.window = None
        self.installed = False
        # SLA config, read by the diagnostic via sla_mix_logits.
        self.sla = False
        self.sla_alpha = 0.3
        self.sla_window = None
        # ---- instrumentation ----
        self.vsv_norm_mean = None
        self.lambda_sim_sum = 0.0
        self.lambda_sim_count = 0
        self.n_steered_forwards = 0
        # per-forward state
        self.current_is_prefill = False
        self._steered_this_forward = False

    def set_vsv(self, vsv: torch.Tensor | None) -> None:
        """Arm (or clear) the steering vector; records its mean per-layer norm."""
        if vsv is None:
            self.vsv = None
            self.vsv_norm_mean = None
            return
        self.vsv = vsv.float()
        self.vsv_norm_mean = float(self.vsv.norm(dim=-1).mean())

    def is_active(self, layer_idx: int) -> bool:
        """Whether this layer steers: a VSV is set, ``lam`` is nonzero, in window."""
        if self.vsv is None or self.lam == 0.0:
            return False
        if self.window is None:
            return True
        start, end = self.window
        return start <= layer_idx <= end

    def begin_forward(self, is_prefill: bool) -> None:
        self.current_is_prefill = bool(is_prefill)
        self._steered_this_forward = False

    def record_steer(self, last_lambda_sim: float) -> None:
        """Count this forward once and accumulate the prefill last-position weight."""
        if not self._steered_this_forward:
            self.n_steered_forwards += 1
            self._steered_this_forward = True
        if self.current_is_prefill:
            self.lambda_sim_sum += float(last_lambda_sim)
            self.lambda_sim_count += 1

    def lambda_sim_mean(self) -> float:
        if self.lambda_sim_count == 0:
            return float("nan")
        return self.lambda_sim_sum / self.lambda_sim_count


# ------------------------------------------------------------------ install


def _extract_memvr_buffer(forward):
    """Return the ``MemVRBuffer`` captured by a MemVR mlp wrapper, else ``None``.

    Structural mutual-exclusion probe: MemVR and VISTA both wrap ``mlp.forward``,
    and both steering at once is out of scope for this phase. Reading MemVR's
    buffer from its wrapper closure lets VISTA detect, at steer time, that MemVR
    is armed on the same forward and fail loudly rather than silently compose two
    interventions. Degrades to ``None`` (no guard) if the wrapper shape changes;
    the pipelines enforce exclusivity at the wiring level too.
    """
    func = getattr(forward, "__func__", forward)
    if getattr(func, "__name__", "") != "_memvr_mlp_forward":
        return None
    freevars = getattr(getattr(func, "__code__", None), "co_freevars", ())
    closure = getattr(func, "__closure__", None) or ()
    for name, cell in zip(freevars, closure):
        if name == "buffer":
            return cell.cell_contents
    return None


def _make_vista_mlp_forward(original_forward, buffer, layer_idx, memvr_buffer):
    """Wrap ``mlp.forward`` to steer the output when armed (patch-all pattern)."""

    def _vista_mlp_forward(self, *args, **kwargs):
        active = buffer.is_active(layer_idx)
        # Guard BEFORE calling the inner forward: a MemVR wrapper disarms itself
        # the moment it injects, so a post-call check would miss the collision.
        if active and memvr_buffer is not None and memvr_buffer.armed_layer is not None:
            raise RuntimeError(
                "MemVR and VISTA are both armed on the same mlp; the two "
                "steering wrappers must not run in the same forward in this "
                "phase (enforce mutual exclusivity in the wiring)."
            )
        x = original_forward(*args, **kwargs)
        if not active:
            return x  # exact short-circuit: no VSV / lam 0 / out of window
        v = buffer.vsv[layer_idx]
        x_new = vista_steer(x, v, buffer.lam)
        lambda_sim = vista_adaptive_factor(x, v)
        buffer.record_steer(float(lambda_sim.reshape(-1)[-1]))
        return x_new

    return _vista_mlp_forward


class VistaInstallation:
    """Handle returned by ``install_vista_hooks``; ``remove()`` undoes everything."""

    def __init__(self, hook_handles, original_mlp_forwards, buffer):
        self._hook_handles = hook_handles
        self._original_mlp_forwards = original_mlp_forwards
        self._buffer = buffer

    def remove(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        for mlp, original in self._original_mlp_forwards:
            mlp.forward = original
        self._hook_handles = []
        self._original_mlp_forwards = []
        self._buffer.installed = False


def install_vista_hooks(model_wrapper, buffer: VistaBuffer) -> VistaInstallation:
    """Install VISTA on a loaded model; return a removable handle.

    A layer-0 forward pre-hook marks the forward boundary (prefill detection by
    ``seq_len > 1``, for the instrumentation) and a per-layer ``mlp.forward``
    wrapper steers the output when ``buffer`` is armed. Reads the buffer at call
    time; no VSV means every layer is a no-op.
    """
    model = model_wrapper._model  # noqa: SLF001
    decoder = decoder_of(model)
    layers = decoder.layers

    hook_handles = []
    original_mlp_forwards = []

    def _layer0_pre_hook(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        buffer.begin_forward(hidden.shape[1] > 1)

    hook_handles.append(
        layers[0].register_forward_pre_hook(_layer0_pre_hook, with_kwargs=True)
    )

    for i, layer in enumerate(layers):
        mlp = _mlp_module_of(layer)
        original_forward = mlp.forward
        memvr_buffer = _extract_memvr_buffer(original_forward)
        original_mlp_forwards.append((mlp, original_forward))
        mlp.forward = MethodType(
            _make_vista_mlp_forward(original_forward, buffer, i, memvr_buffer), mlp
        )

    buffer.installed = True
    return VistaInstallation(hook_handles, original_mlp_forwards, buffer)
