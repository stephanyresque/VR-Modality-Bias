from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Vocabulary", "load_vocabulary"]


@dataclass(frozen=True)
class Vocabulary:
    name: str
    categories: tuple[str, ...]
    synonyms: dict[str, tuple[str, ...]]
    synonym_to_category: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(
            self,
            "synonyms",
            {cat: tuple(syns) for cat, syns in self.synonyms.items()},
        )
        self._reject_unknown_categories()
        self._reject_ambiguous_synonyms()
        index: dict[str, str] = {}
        for category in self.categories:
            index[category] = category
            for synonym in self.synonyms.get(category, ()):
                index[synonym] = category
        object.__setattr__(self, "synonym_to_category", index)

    def _reject_unknown_categories(self) -> None:
        unknown = sorted(set(self.synonyms) - set(self.categories))
        if unknown:
            raise ValueError(
                f"vocabulary {self.name!r}: synonym map has {len(unknown)} key(s) "
                f"that are not declared categories: {unknown}."
            )

    def _reject_ambiguous_synonyms(self) -> None:
        owners: dict[str, list[str]] = defaultdict(list)
        for category, syns in self.synonyms.items():
            for synonym in syns:
                owners[synonym].append(category)
        ambiguous = {s: sorted(c) for s, c in owners.items() if len(c) > 1}
        if ambiguous:
            listed = ", ".join(
                f"{s!r} under {cats}" for s, cats in sorted(ambiguous.items())
            )
            raise ValueError(
                f"vocabulary {self.name!r}: {len(ambiguous)} synonym(s) claimed by "
                f"more than one category: {listed}."
            )


def load_vocabulary(path: Path) -> Vocabulary:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = sorted({"name", "categories", "synonyms"} - set(data))
    if missing:
        raise ValueError(f"{path}: vocabulary file is missing key(s): {missing}.")
    try:
        return Vocabulary(
            name=str(data["name"]),
            categories=tuple(data["categories"]),
            synonyms={str(k): tuple(v) for k, v in data["synonyms"].items()},
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
