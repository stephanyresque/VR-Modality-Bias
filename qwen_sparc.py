# Uso (rodar a partir da raiz do repositorio, na branch qwen-sparc):
#   python qwen_sparc.py
#   python qwen_sparc.py --image-id 000000000139 --max-new-tokens 512

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import transformers
from PIL import Image

try:
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.experiment.sparc import (
        SparcHyperparams,
        enable_sparc,
        probe_image_token_index,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.experiment.sparc import (
        SparcHyperparams,
        enable_sparc,
        probe_image_token_index,
    )


MODEL_KEY = "qwen2.5-vl-7b"
DEFAULT_PROMPT = "Describe the image in a long, detailed paragraph."

SPARC_OFFICIAL_COCO = dict(
    alpha=1.1,
    beta=0.1,
    tau=1.5,
    selected_layer=20,
    se_layers=(0, 31),
)


def _find_image(image_id: str) -> Path:
    images_dir = Path("data/processed/mscoco_baseline/images")
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    candidates = sorted(images_dir.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em {images_dir}. "
            "Rode a partir da raiz do repositorio."
        )
    print(f"[aviso] {image_id} nao encontrada; usando {candidates[0].name}")
    return candidates[0]


def audit_indexing(wrapper, image: Image.Image, prompt: str) -> None:
    print("=" * 78)
    print("AUDITORIA DE INDEXACAO - Qwen-2.5-VL")
    print("=" * 78)

    image_token_id = int(wrapper._model.config.image_token_id)
    image_token_index, input_len, num_image_patches = probe_image_token_index(
        wrapper, image, prompt
    )

    messages = wrapper._build_messages(prompt, image)
    prompt_text = wrapper._processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = wrapper._processor(
        text=[prompt_text], images=[image], return_tensors="pt"
    )
    input_ids = inputs["input_ids"][0]

    block = input_ids[image_token_index : image_token_index + num_image_patches]
    n_match = int((block == image_token_id).sum())

    print(f"  config.image_token_id : {image_token_id}")
    print(f"  image_token_index     : {image_token_index}")
    print(f"  num_image_patches     : {num_image_patches}")
    print(f"  input_len (sem patches): {input_len}")
    print(f"  total seq length      : {int(input_ids.shape[-1])}")
    print(
        f"  posicoes [{image_token_index}, "
        f"{image_token_index + num_image_patches}) que sao image_token: "
        f"{n_match}/{num_image_patches}"
    )
    if n_match == num_image_patches:
        print("  VEREDITO: PASS - 100% tokens de imagem. "
              "A degeneracao NAO vem de indexacao no Qwen.")
    else:
        print(f"  VEREDITO: {num_image_patches - n_match} posicoes NAO sao "
              "tokens de imagem (inesperado para o Qwen).")
    print()


def generate(wrapper, image: Image.Image, prompt: str, max_new_tokens: int,
             sparc: bool) -> str:
    gen_kwargs = dict(do_sample=False, num_beams=1, repetition_penalty=1.0)
    if not sparc:
        return wrapper.generate_caption(
            image, prompt, max_new_tokens=max_new_tokens, seed=42,
            generation_kwargs=gen_kwargs,
        )
    hparams = SparcHyperparams(**SPARC_OFFICIAL_COCO)
    with enable_sparc(wrapper, hparams=hparams, probe_image=image, prompt=prompt):
        return wrapper.generate_caption(
            image, prompt, max_new_tokens=max_new_tokens, seed=42,
            generation_kwargs=gen_kwargs,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", default="000000000139")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    print("=" * 78)
    print("DEBUG - SPARC no Qwen-2.5-VL-7B (reproducao isolada)")
    print("=" * 78)
    print(f"  transformers version : {transformers.__version__}")
    print(f"  torch version        : {torch.__version__}")
    print(f"  CUDA disponivel      : {torch.cuda.is_available()}")
    print(f"  model_key            : {MODEL_KEY}")
    print(f"  SPARC (oficial COCO) : {SPARC_OFFICIAL_COCO}")
    print(f"  decoding             : greedy (do_sample=False, num_beams=1)")
    print("=" * 78)
    print()

    image_path = _find_image(args.image_id)
    image = Image.open(image_path).convert("RGB")
    print(f"imagem: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = build_model(MODEL_KEY)
    print(f"carregando {MODEL_KEY} em {device}...")
    wrapper.load(device)
    print(f"carregado. n_layers={wrapper.n_layers}")
    print()

    audit_indexing(wrapper, image, args.prompt)

    print("=" * 78)
    print("LEGENDA SEM SPARC (baseline)")
    print("=" * 78)
    base = generate(wrapper, image, args.prompt, args.max_new_tokens, sparc=False)
    print(base)
    print()

    print("=" * 78)
    print("LEGENDA COM SPARC (config oficial COCO)")
    print("=" * 78)
    on = generate(wrapper, image, args.prompt, args.max_new_tokens, sparc=True)
    print(on)
    print()

    print("=" * 78)
    print("ONDE INVESTIGAR (acoplamento da atencao x versao do transformers)")
    print("=" * 78)
    print("  src/vr_modality_bias/utils/attn.py:")
    print("    - forward_qwen25vl(...)")
    print("    - apply_multimodal_rotary_pos_emb")
    print("    - add_custom_attention_layers(...)")
    print("  src/vr_modality_bias/experiment/forced_decoding.py:")
    print("    - manejo de past_key_values / Cache no loop de geracao")
    print()
    print(f"  transformers em uso: {transformers.__version__}")
    print("  Indexacao confirmada OK acima -> foco na atencao/cache.")
    print("=" * 78)


if __name__ == "__main__":
    main()