from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_GOLDEN_SET_PATH = Path(__file__).parent.parent / "data" / "golden_set.yaml"


@dataclass(frozen=True)
class GoldenItem:
    """A single hand-curated eval item.

    Frozen so items can be used as parametrize ids and so a test can't
    mutate the dataset mid-run.
    """

    id: str
    question: str
    expected: str
    difficulty: str
    category: str
    tags: tuple[str, ...] = field(default_factory=tuple)


_REQUIRED_FIELDS = {"id", "question", "expected", "difficulty", "category"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_golden_set(path: Path | str = DEFAULT_GOLDEN_SET_PATH) -> list[GoldenItem]:
    """Load and validate the golden set.

    Validation is intentionally strict: a malformed dataset should fail loud
    at load time.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Golden set at {path} must be a non-empty list")

    items: list[GoldenItem] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"Item {i} is not a mapping: {row!r}")
        missing = _REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Item {i} ({row.get('id')!r}) missing fields: {sorted(missing)}")
        if row["difficulty"] not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Item {row['id']!r} has invalid difficulty {row['difficulty']!r}; "
                f"expected one of {sorted(_VALID_DIFFICULTIES)}"
            )
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate item id: {row['id']!r}")
        seen_ids.add(row["id"])
        items.append(
            GoldenItem(
                id=row["id"],
                question=row["question"],
                expected=row["expected"],
                difficulty=row["difficulty"],
                category=row["category"],
                tags=tuple(row.get("tags") or ()),
            )
        )
    return items
