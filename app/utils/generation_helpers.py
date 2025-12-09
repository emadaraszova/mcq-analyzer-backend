"""Shared helpers for question generation and demographic batching."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Match clinical scenarios delimited by XXX ... XXX (multiline, case-insensitive).
SCENARIO_PATTERN = re.compile(r"XXX[\s\S]*?XXX", re.IGNORECASE)


def count_scenarios(text: str) -> int:
    """Count XXX-delimited scenarios in text."""
    return len(SCENARIO_PATTERN.findall(text))


def trim_to_n_questions(text: str, n: int) -> str:
    """Trim concatenated output so it contains at most ``n`` XXX blocks."""
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        out_lines.append(line)
        if "XXX" in line and count_scenarios("".join(out_lines)) >= n:
            break
    return "".join(out_lines)


def pick_filled_category(
    demo: Optional[Dict[str, List[Dict[str, Any]]]],
) -> Optional[str]:
    """Return the first demographic category that has at least one row."""
    if not demo:
        return None
    for cat in ["Sex", "Ethnicity", "Age"]:
        if demo.get(cat, []):
            return cat
    return None


def category_constraint_phrase(category: str, label: str) -> str:
    """Human-readable phrase describing a subgroup constraint."""
    if category == "Sex":
        lower = label.strip().lower()
        if lower in ("female", "woman", "women"):
            return "patients who are women"
        if lower in ("male", "man", "men"):
            return "patients who are men"
        return f"patients whose sex is {label}"
    if category == "Ethnicity":
        return f"patients whose ethnicity is {label}"
    if category == "Age":
        return f"patients aged {label}"
    return f"patients matching: {label}"


def build_initial_prompt(base_message: str, count: int) -> str:
    """
    Build the first user message when there's NO demographic constraint.
    FE no longer includes the number, so BE appends it here.
    """
    return (
        f"{base_message.strip()}\n\n"
        f"Generate exactly {count} question(s) (in batches of max 10).\n"
    )


def build_group_prompt(base_message: str, category: str, label: str, count: int) -> str:
    """Build the first user message for a demographic batch."""
    group_phrase = category_constraint_phrase(category, label)
    return (
        f"{base_message.strip()}\n\n"
        f"Generate exactly {count} question(s) for {group_phrase}.\n"
    )


def build_group_followup(
    message: str,
    remaining: int,
    category: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Follow-up starts with the original initial `message` and then adds only the
    information about the remaining questions (no extra reminders).
    """
    if category and label:
        group_phrase = category_constraint_phrase(category, label)
        return (
            f"{message.strip()}\n\n"
            f"Please continue with the remaining {remaining} question(s) for {group_phrase}."
        )
    return (
        f"{message.strip()}\n\n"
        f"Please continue with the remaining {remaining} question(s)."
    )
