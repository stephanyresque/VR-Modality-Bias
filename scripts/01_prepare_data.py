#!/usr/bin/env python
"""Download MSCOCO val2017 and stage the baseline image subset."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger

MSCOCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
MSCOCO_VAL2017_FILENAME = "val2017.zip"
RAW_ROOT = Path("data/raw/mscoco")


def download_file(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Stream-download ``url`` to ``dest``. Idempotent unless ``overwrite``."""
    log = get_logger(__name__)
    if dest.exists() and not overwrite:
        log.info(
            "Skipping download — already present: %s (%.1f MB)",
            dest,
            dest.stat().st_size / 1e6,
        )
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with tmp.open("wb") as fh, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def extract_subset(
    zip_path: Path,
    out_dir: Path,
    n_images: int,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Extract the first ``n_images`` ``.jpg`` entries (lex order) of ``zip_path``."""
    log = get_logger(__name__)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.endswith(".jpg") and not name.endswith("/")
        )
        if len(names) < n_images:
            raise RuntimeError(
                f"Archive {zip_path} has only {len(names)} images but {n_images} requested."
            )
        chosen = names[:n_images]
        extracted: list[Path] = []
        for name in chosen:
            out_path = out_dir / Path(name).name
            if out_path.exists() and not overwrite:
                extracted.append(out_path)
                continue
            with zf.open(name) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
    log.info("Extracted %d image(s) to %s", len(extracted), out_dir)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download the zip and re-extract images even if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override dataset.n_images for this run (useful for smoke tests).",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger(__name__)

    cfg = load_config(args.config)
    n_images = int(args.limit) if args.limit is not None else int(cfg["dataset"]["n_images"])
    images_dir = Path(cfg["dataset"]["images_dir"])
    zip_path = RAW_ROOT / MSCOCO_VAL2017_FILENAME

    log.info(
        "Preparing %d image(s) into %s (raw: %s)", n_images, images_dir, zip_path
    )
    download_file(MSCOCO_VAL2017_URL, zip_path, overwrite=args.overwrite)
    paths = extract_subset(zip_path, images_dir, n_images, overwrite=args.overwrite)
    log.info("Done. %d image(s) ready under %s.", len(paths), images_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
