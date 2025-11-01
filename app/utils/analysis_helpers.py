"""Shared helpers for clinical scenario analysis."""

from __future__ import annotations

import re
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar

# Regex: match text delimited by XXX ... XXX (multiline, case-insensitive).
SCENARIO_PATTERN = re.compile(r"XXX([\s\S]*?)XXX", re.IGNORECASE)

T = TypeVar("T")


def extract_scenarios(text: str) -> List[Tuple[int, str]]:
    """Extract scenarios delimited by ``XXX ... XXX`` from raw text.

    Returns a list of (original_index, scenario_text) tuples in original order.
    """
    scenarios: List[Tuple[int, str]] = []
    for i, match in enumerate(SCENARIO_PATTERN.finditer(text)):
        scenario = match.group(1).strip()
        if scenario:
            scenarios.append((i, scenario))
    return scenarios


def chunk(seq: Sequence[T] | Iterable[T], size: int) -> Iterator[List[T]]:
    """Yield successive chunks of at most ``size`` items from a sequence/iterable."""
    if size <= 0:
        raise ValueError("size must be > 0")
    buf: List[T] = []
    for item in seq:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
