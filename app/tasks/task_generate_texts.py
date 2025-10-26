"""Question generation orchestration and demographic batching (with logging)."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService
from app.services.einfra_service import EInfraService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Match clinical scenarios delimited by XXX ... XXX (multiline, case-insensitive).
SCENARIO_PATTERN = re.compile(r"XXX[\s\S]*?XXX", re.IGNORECASE)


def count_scenarios(text: str) -> int:
    """Count XXX-delimited scenarios in text."""
    return len(SCENARIO_PATTERN.findall(text))


def trim_to_n_questions(text: str, n: int) -> str:
    """Trim concatenated output so it contains at most ``n`` XXX blocks.

    This is used when the model overgenerates. We keep appending original
    lines until we’ve accumulated ``n`` complete XXX…XXX blocks.
    """
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        out_lines.append(line)
        if "XXX" in line and count_scenarios("".join(out_lines)) >= n:
            break
    return "".join(out_lines)


def select_service(model_name: str):
    """Return the provider service based on the model prefix.

    Notes:
        * We keep it simple and key off the prefix to select a provider.
        * Printing is deliberate here to surface provider decisions in logs.
    """
    if model_name.startswith("llama"):
        print(
            f"[generate] select_service → EInfraService (model='{model_name}')",
            flush=True,
        )
        return EInfraService()
    if model_name.startswith("gemini"):
        print(
            f"[generate] select_service → GeminiService (model='{model_name}')",
            flush=True,
        )
        return GeminiService()
    if model_name.startswith("gpt") or model_name == "chatgpt-4o-latest":
        print(
            f"[generate] select_service → OpenAIService (model='{model_name}')",
            flush=True,
        )
        return OpenAIService()
    raise ValueError(
        "Invalid model specified: {model_name}. Supported: gemini*, gpt-*."
    )


# ---------------------------------------------------------------------------
# Demographic helpers
# ---------------------------------------------------------------------------


def pick_filled_category(
    demo: Optional[Dict[str, List[Dict[str, Any]]]],
) -> Optional[str]:
    """Return the first demographic category that has at least one row.

    We look in a fixed priority order so behavior is deterministic if more
    than one category is filled.
    """
    if not demo:
        return None
    for cat in ["Gender", "Ethnicity", "Age"]:
        if demo.get(cat, []):
            return cat
    return None


def category_constraint_phrase(category: str, label: str) -> str:
    """Human-readable phrase describing a subgroup constraint.

    The phrasing is meant for prompts—short, clear, and unambiguous.
    """
    if category == "Gender":
        lower = label.strip().lower()
        if lower in ("female", "woman", "women"):
            return "patients who are women"
        if lower in ("male", "man", "men"):
            return "patients who are men"
        return f"patients whose gender is {label}"
    if category == "Ethnicity":
        return f"patients whose ethnicity is {label}"
    if category == "Age":
        return f"patients aged {label}"
    return f"patients matching: {label}"


def build_group_prompt(base_message: str, category: str, label: str, count: int) -> str:
    """Build the first user message for a demographic batch.

    Adds an exact-count requirement and a clear subgroup hint.
    """
    group_phrase = category_constraint_phrase(category, label)
    return (
        f"{base_message.strip()}\n\n"
        f"Generate exactly {count} question(s) for {group_phrase}.\n"
    )


def build_group_followup(
    remaining: int, category: Optional[str], label: Optional[str]
) -> str:
    """Build a follow-up message for the same subgroup.

    Used when we need to continue generation in incremental rounds.
    """
    if category and label:
        group_phrase = category_constraint_phrase(category, label)
        return (
            f"Please continue with the remaining {remaining} question(s) "
            f"for {group_phrase}."
        )
    return f"Please continue with the remaining {remaining} question(s)."


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def _single_run_generate_exact(
    service: Any,
    model: str,
    base_message: str,
    exact_target: int,
    group_category: Optional[str] = None,
    group_label: Optional[str] = None,
) -> Tuple[str, int]:
    """Generate up to ``exact_target`` XXX-delimited questions in one run.

    Strategy:
        1) Send an initial prompt (optionally constrained by demographics).
        2) Count XXX blocks in the reply.
        3) If under target, keep asking the model to continue until:
           - target is reached, or
           - no progress is observed, or
           - the model returns empty/whitespace.
        4) If we overshoot, trim back to the requested count.

    Returns:
        (full_text, count_of_XXX_blocks)
    """
    run_id = str(uuid.uuid4())
    print(
        f"[generate:{run_id}] START exact_target={exact_target} model={model} "
        f"group=({group_category}={group_label})",
        flush=True,
    )

    # --- Compose the first user message (with or without subgroup constraint) ---
    if group_category and group_label:
        user_first_message = build_group_prompt(
            base_message, group_category, group_label, exact_target
        )
        print(
            f"[generate:{run_id}] First message WITH group constraint:\n"
            f"---8<---\n{user_first_message}\n---8<---",
            flush=True,
        )
    else:
        user_first_message = base_message
        print(
            f"[generate:{run_id}] First message (base prompt only):\n"
            f"---8<---\n{user_first_message}\n---8<---",
            flush=True,
        )

    # Minimal chat session that all providers accept.
    session = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_first_message},
    ]

    class _UserMessageShim:
        """Tiny adapter to match `service.get_response` signature across providers."""

        def __init__(self, message: str, model: str) -> None:
            self.message = message
            self.model = model

    # --- First round call ---
    print(f"[generate:{run_id}] Calling service.get_response(first)...", flush=True)
    first_reply = service.get_response(
        _UserMessageShim(user_first_message, model), session
    )
    accumulated_text = first_reply or ""
    session.append({"role": "assistant", "content": first_reply})

    first_count = count_scenarios(accumulated_text)
    print(
        f"[generate:{run_id}] First reply length={len(accumulated_text)} "
        f"scenarios_found={first_count}",
        flush=True,
    )

    # If there are no XXX delimiters, we can't reliably count → return raw text.
    if first_count == 0:
        print(
            f"[generate:{run_id}] No 'XXX' delimiters found → returning raw text.",
            flush=True,
        )
        return accumulated_text, 0

    # Already hit or exceeded the target on the first round.
    if first_count >= exact_target:
        if first_count > exact_target:
            print(
                f"[generate:{run_id}] Overshot ({first_count}>{exact_target}) → trimming.",
                flush=True,
            )
            accumulated_text = trim_to_n_questions(accumulated_text, exact_target)
        print(f"[generate:{run_id}] DONE (hit target in first round).", flush=True)
        return accumulated_text, min(first_count, exact_target)

    # --- Progress-driven loop (no hard cap) ---
    # Continues until target is reached or progress stalls.
    last_count = first_count
    while True:
        current_count = count_scenarios(accumulated_text)
        if current_count >= exact_target:
            print(
                f"[generate:{run_id}] Reached target ({current_count}≥{exact_target}).",
                flush=True,
            )
            break

        remaining = exact_target - current_count
        followup = build_group_followup(remaining, group_category, group_label)

        print(
            f"[generate:{run_id}] Asking follow-up (remaining={remaining}): {followup}",
            flush=True,
        )
        session.append({"role": "user", "content": followup})
        assistant_text = service.get_response(
            _UserMessageShim(followup, model), session
        )

        # Stop if the model returns nothing useful (prevents infinite loops).
        if not assistant_text or assistant_text.strip() == "":
            print(
                f"[generate:{run_id}] Empty/whitespace reply → stop (partial).",
                flush=True,
            )
            break

        # Append new content and track it in the conversation history.
        accumulated_text += ("\n\n" if accumulated_text else "") + assistant_text
        session.append({"role": "assistant", "content": assistant_text})

        # Count again after appending; ensure we made progress.
        new_count = count_scenarios(accumulated_text)
        print(
            f"[generate:{run_id}] Chunk len={len(assistant_text)} "
            f"total_scenarios={new_count}",
            flush=True,
        )
        if new_count <= last_count:
            print(
                f"[generate:{run_id}] No progress (last={last_count}, new={new_count}) → break.",
                flush=True,
            )
            break
        last_count = new_count

    # Final safety trim if we overshot the requested count.
    final_count = count_scenarios(accumulated_text)
    if final_count > exact_target:
        print(
            f"[generate:{run_id}] Final overshoot ({final_count}>{exact_target}) → trimming.",
            flush=True,
        )
        accumulated_text = trim_to_n_questions(accumulated_text, exact_target)
        final_count = exact_target

    print(
        f"[generate:{run_id}] END final_count={final_count}/{exact_target}",
        flush=True,
    )
    return accumulated_text, final_count


def generate_texts(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate question generation (single batch or demographic batches).

    Two modes:
        • CASE A – single-batch generation (no demographics provided)
        • CASE B – multi-batch generation according to a demographic distribution
    """
    message: str = payload.get("message", "")
    model: str = payload.get("model", "gpt-4o")
    print(f"[generate] payload keys={list(payload.keys())}", flush=True)
    target: int = int(payload.get("number_of_questions", 1))
    demo: Optional[Dict[str, List[Dict[str, Any]]]] = payload.get("demographicData")

    if not message or not message.strip():
        raise ValueError("Payload 'message' must be a non-empty string.")

    session_id = str(uuid.uuid4())
    print("[generate] ===== NEW REQUEST =====", flush=True)
    print(
        f"[generate] session_id={session_id} model={model} target={target}",
        flush=True,
    )

    service = select_service(model)

    chosen_category = pick_filled_category(demo)
    print(f"[generate:{session_id}] chosen_category={chosen_category}", flush=True)

    # --- CASE A: Simple single-batch generation (no demographics) -------------
    if not chosen_category:
        print(
            f"[generate:{session_id}] Case A (no demographic batching).",
            flush=True,
        )
        print(
            f"[generate:{session_id}] BASE PROMPT:\n---8<---\n{message}\n---8<---",
            flush=True,
        )

        text, actual = _single_run_generate_exact(service, model, message, target)
        status = "completed" if actual >= target else "partial"
        print(
            f"[generate:{session_id}] status={status} actual={actual}/{target}",
            flush=True,
        )

        # If we couldn't detect XXX blocks at all, return whatever the model
        # produced and note why (helps debugging front-end expectations).
        if actual == 0:
            return {
                "status": "completed",
                "session_id": session_id,
                "model": model,
                "message": message,
                "requested_number_of_questions": target,
                "actual_number_of_questions": 0,
                "response": text,
                "note": "No XXX delimiters found; returned initial response.",
            }

        return {
            "status": status,
            "session_id": session_id,
            "model": model,
            "message": message,
            "requested_number_of_questions": target,
            "actual_number_of_questions": actual,
            "response": text,
        }

    # --- CASE B: Multi-batch generation based on demographic distribution -----
    print(f"[generate:{session_id}] Case B (batched by {chosen_category}).", flush=True)
    groups = demo.get(chosen_category, []) if demo else []
    if not groups:
        print(
            f"[generate:{session_id}] groups list empty → fallback to Case A.",
            flush=True,
        )
        text, actual = _single_run_generate_exact(service, model, message, target)
        return {
            "status": "completed",
            "session_id": session_id,
            "model": model,
            "message": message,
            "requested_number_of_questions": target,
            "actual_number_of_questions": actual,
            "response": text,
        }

    # Validate sum of per-group counts matches the requested total.
    total_requested = sum(int(g.get("value", 0)) for g in groups)
    print(
        f"[generate:{session_id}] groups={groups} total_requested={total_requested}",
        flush=True,
    )
    if total_requested != target:
        raise ValueError(
            f"Demographic distribution total ({total_requested}) must equal "
            f"number_of_questions ({target})."
        )

    combined_texts: List[str] = []
    combined_count = 0
    batches_meta: List[Dict[str, Any]] = []

    # Process each demographic group independently and collect results.
    for idx, group in enumerate(groups, start=1):
        label: str = str(group["label"])
        qty: int = int(group["value"])
        print(
            f"[generate:{session_id}] ---- Group {idx}/{len(groups)}: "
            f"{chosen_category}='{label}', qty={qty}",
            flush=True,
        )

        # Log the effective prompt used for transparency/debugging.
        eff_prompt = build_group_prompt(message, chosen_category, label, qty)
        print(
            f"[generate:{session_id}] Effective prompt for group:\n"
            f"---8<---\n{eff_prompt}\n---8<---",
            flush=True,
        )

        batch_text, batch_actual = _single_run_generate_exact(
            service=service,
            model=model,
            base_message=message,
            exact_target=qty,
            group_category=chosen_category,
            group_label=label,
        )
        print(
            f"[generate:{session_id}] Group done {chosen_category}='{label}': "
            f"actual={batch_actual}/{qty} (len={len(batch_text or '')})",
            flush=True,
        )

        # Per-group metadata is returned so the UI can display breakdowns.
        batches_meta.append(
            {
                "index": idx,
                "category": chosen_category,
                "label": label,
                "requested": qty,
                "actual": batch_actual,
                "prompt_used": eff_prompt.strip(),
                "text": batch_text,
            }
        )

        # Add a human-readable header before each subgroup block in the final text.
        header = f"### Group: {chosen_category}='{label}' ({batch_actual}/{qty})\n\n"
        combined_texts.append(header + (batch_text or ""))
        combined_count += batch_actual

    # Merge groups and (if needed) trim overshoot.
    combined_response = "\n\n".join(t for t in combined_texts if t.strip())

    total_after_concat = count_scenarios(combined_response)
    print(
        f"[generate:{session_id}] After concat scenarios={total_after_concat}",
        flush=True,
    )
    if total_after_concat > target:
        print(
            f"[generate:{session_id}] Overshoot after concat → trimming to {target}.",
            flush=True,
        )
        combined_response = trim_to_n_questions(combined_response, target)

    status = "completed" if combined_count >= target else "partial"
    print(
        f"[generate:{session_id}] DONE status={status} "
        f"total_actual={combined_count}/{target}",
        flush=True,
    )

    return {
        "status": status,
        "session_id": session_id,
        "model": model,
        "message": message,
        "requested_number_of_questions": target,
        "actual_number_of_questions": min(combined_count, target),
        "response": combined_response,
        "note": f"Batched by {chosen_category}.",
        "batches": batches_meta,
    }
