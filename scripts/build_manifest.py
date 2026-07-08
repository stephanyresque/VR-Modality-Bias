#!/usr/bin/env python
"""Build ``manifest.jsonl`` from the prepared image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from vr_modality_bias.data.manifests import ImageRecord, write_manifest
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records (useful for smoke tests).",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger(__name__)

    cfg = load_config(args.config)
    images_dir = Path(cfg["dataset"]["images_dir"])
    manifest_path = Path(cfg["dataset"]["manifest_path"])
    source = str(cfg["dataset"]["name"])
    expected_n = int(cfg["dataset"]["n_images"])

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}. "
            "Run scripts/prepare_data.py first."
        )

    if manifest_path.exists() and not args.overwrite:
        log.info(
            "Manifest already exists: %s (pass --overwrite to regenerate).",
            manifest_path,
        )
        return 0

    files = sorted(p for p in images_dir.glob("*.jpg") if p.is_file())
    if args.limit is not None:
        files = files[: args.limit]

    records: list[ImageRecord] = []
    for path in files:
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            ImageRecord(
                image_id=path.stem,
                file_name=path.name,
                width=int(width),
                height=int(height),
                source=source,
            )
        )

    n = write_manifest(records, manifest_path)
    log.info("Wrote %d record(s) to %s.", n, manifest_path)
    if args.limit is None and n != expected_n:
        log.warning(
            "Wrote %d records but config.dataset.n_images=%d. "
            "Re-run scripts/prepare_data.py?",
            n,
            expected_n,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
