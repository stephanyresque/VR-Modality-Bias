from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "COMPONENT_TYPES",
    "ObjectAnnotation",
    "QuestionAnnotation",
    "QuestionComponent",
    "iter_object_annotations",
    "iter_question_annotations",
    "read_object_annotations",
    "read_question_annotations",
    "write_object_annotations",
    "write_question_annotations",
]

COMPONENT_TYPES: tuple[str, ...] = ("existence", "count", "direction")


@dataclass(frozen=True)
class QuestionComponent:
    component_type: str
    target: str
    answer: str | int

    def __post_init__(self) -> None:
        if self.component_type not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component_type {self.component_type!r}. "
                f"Valid types: {list(COMPONENT_TYPES)}."
            )


@dataclass(frozen=True)
class ObjectAnnotation:
    image_id: str
    objects: tuple[str, ...]
    counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True)
class QuestionAnnotation:
    image_id: str
    question_id: str
    question_text: str
    components: tuple[QuestionComponent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            tuple(
                component
                if isinstance(component, QuestionComponent)
                else QuestionComponent(**component)
                for component in self.components
            ),
        )


_Annotation = ObjectAnnotation | QuestionAnnotation


def _write_annotations(records: Iterable[_Annotation], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n += 1
    return n


def _iter_annotations(
    path: Path, record_cls: type[_Annotation]
) -> Iterator[_Annotation]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield record_cls(**data)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: malformed record on line {lineno}: {exc}"
                ) from exc


def write_object_annotations(records: Iterable[ObjectAnnotation], path: Path) -> int:
    return _write_annotations(records, path)


def read_object_annotations(path: Path) -> list[ObjectAnnotation]:
    return list(iter_object_annotations(path))


def iter_object_annotations(path: Path) -> Iterator[ObjectAnnotation]:
    return _iter_annotations(path, ObjectAnnotation)


def write_question_annotations(
    records: Iterable[QuestionAnnotation], path: Path
) -> int:
    return _write_annotations(records, path)


def read_question_annotations(path: Path) -> list[QuestionAnnotation]:
    return list(iter_question_annotations(path))


def iter_question_annotations(path: Path) -> Iterator[QuestionAnnotation]:
    return _iter_annotations(path, QuestionAnnotation)
