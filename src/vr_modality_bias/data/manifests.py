"""Manifest read/write --- one JSON Lines record per image."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["ImageRecord", "iter_manifest", "read_manifest", "write_manifest"]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    file_name: str
    width: int
    height: int
    source: str
    dataset: str | None = None
    scene_id: str | None = None
    frame_index: int | None = None

    def __post_init__(self) -> None:
        if "/" in self.image_id or "\\" in self.image_id:
            raise ValueError(
                f"image_id {self.image_id!r} contains a path separator. The "
                "diagnostic names its artefacts '{image_id}__A.h5', so a "
                "separator would resolve to a subdirectory that does not "
                "exist and the write would fail far from the cause."
            )


def write_manifest(records: Iterable[ImageRecord], path: Path) -> int:
    """Write ``records`` as JSON Lines at ``path``. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_manifest(path: Path) -> list[ImageRecord]:
    """Load all records from a JSON Lines manifest at ``path``."""
    return list(iter_manifest(path))


def iter_manifest(path: Path) -> Iterator[ImageRecord]:
    """Iterate over records in the manifest at ``path`` lazily."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield ImageRecord(**data)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"{path}: malformed record on line {lineno}: {exc}"
                ) from exc
