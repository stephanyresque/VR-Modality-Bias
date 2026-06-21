"""Idempotent COCO val2017 annotation download."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from pyprojroot import here
from tqdm import tqdm

try:
    from vr_modality_bias.utils.logging import configure_logging, get_logger
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))
    from src.vr_modality_bias.utils.logging import configure_logging, get_logger


COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
DEFAULT_TARGET_DIR = Path("data/processed/mscoco_baseline/annotations")
EXTRACTED_FILES = ("instances_val2017.json", "captions_val2017.json")

__all__ = [
    "COCO_ANNOTATIONS_URL",
    "DEFAULT_TARGET_DIR",
    "EXTRACTED_FILES",
    "ensure_coco_annotations",
]


def _download_zip(url: str, dest: Path, *, overwrite: bool) -> Path:
    """Stream-download ``url`` to ``dest``. Idempotent unless ``overwrite``."""
    log = get_logger(__name__)
    if dest.exists() and not overwrite:
        log.info(
            "Zip already present: %s (%.1f MB). Skipping download.",
            dest, dest.stat().st_size / 1e6,
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


def ensure_coco_annotations(
    target_dir: Path = DEFAULT_TARGET_DIR,
    *,
    overwrite: bool = False,
    cleanup_zip: bool = True,
) -> dict[str, Path]:
    """Download and extract COCO val2017 annotations idempotently.

    Args:
        target_dir: Where to put the extracted JSON files.
        overwrite: Re-download and re-extract even if files exist.
        cleanup_zip: Delete the zip after extraction (default True; the
            files we keep total < 30 MB, the zip is > 200 MB).

    Returns:
        Dict mapping the file name to its on-disk path.
    """
    log = get_logger(__name__)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: target_dir / name for name in EXTRACTED_FILES}
    all_present = all(p.exists() for p in paths.values())
    if all_present and not overwrite:
        for name, p in paths.items():
            log.info("Annotation present: %s (%.1f MB)", p, p.stat().st_size / 1e6)
        return paths

    zip_path = target_dir / "annotations_trainval2017.zip"
    _download_zip(COCO_ANNOTATIONS_URL, zip_path, overwrite=overwrite)

    log.info("Extracting %s to %s", list(EXTRACTED_FILES), target_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zip_names = set(zf.namelist())
        for name in EXTRACTED_FILES:
            zip_entry = f"annotations/{name}"
            if zip_entry not in zip_names:
                raise RuntimeError(
                    f"{zip_entry!r} not in zip; available example entries: "
                    f"{sorted(zip_names)[:5]}"
                )
            out_path = paths[name]
            with zf.open(zip_entry) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            log.info("Extracted %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    if cleanup_zip:
        zip_path.unlink()
        log.info("Removed zip %s", zip_path)

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir", type=Path, default=DEFAULT_TARGET_DIR,
        help="Where to place the extracted JSONs.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and re-extract even if the files are already present.",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Don't delete the zip after extraction (default: delete).",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_coco_annotations(
        args.target_dir,
        overwrite=args.overwrite,
        cleanup_zip=not args.keep_zip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
