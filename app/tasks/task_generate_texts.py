import re
import uuid
from typing import Dict, Any, List, Optional, Tuple
from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCENARIO_PATTERN = re.compile(r'XXX[\s\S]*?XXX', re.IGNORECASE)

def count_scenarios(text: str) -> int:
    return len(SCENARIO_PATTERN.findall(text))

def trim_to_n_questions(text: str, n: int) -> str:
    """
    Keep appending lines until we have at least n `XXX...XXX` blocks,
    then stop. This preserves the original order and avoids cutting
    in the middle of a block.
    """
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        out_lines.append(line)
        if 'XXX' in line:
            if count_scenarios(''.join(out_lines)) >= n:
                break
    return ''.join(out_lines)

def select_service(m: str):
    """
    Pick the model provider based on the model name.
    """
    if m.startswith("gemini"):
        print(f"[generate] select_service → GeminiService (model='{m}')", flush=True)
        return GeminiService()
    if m.startswith("gpt") or m == "chatgpt-4o-latest":
        print(f"[generate] select_service → OpenAIService (model='{m}')", flush=True)
        return OpenAIService()
    raise ValueError(f"Invalid model specified: {m}. Supported: gemini*, gpt-*.")

# ---------------------------------------------------------------------------
# Demographic helpers
# ---------------------------------------------------------------------------

def pick_filled_category(demo: Optional[Dict[str, List[Dict[str, Any]]]]) -> Optional[str]:
    """
    Return the first demographic category that has at least one row filled,
    or None if no categories are provided.
    """
    if not demo:
        return None
    for cat in ["Gender", "Ethnicity", "Age"]:
        rows = demo.get(cat, [])
        if len(rows) > 0:
            return cat
    return None

def category_constraint_phrase(category: str, label: str) -> str:
    """
    Produce a short, unambiguous phrase describing the group constraint.
    This is used to condition the model for a specific demographic subgroup.
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
    """
    Build the first user message for a demographic batch.
    Adds an exact count requirement and a clear subgroup hint.
    """
    group_phrase = category_constraint_phrase(category, label)
    return (
        f"{base_message.strip()}\n\n"
        f"Generate exactly {count} question(s) for {group_phrase}.\n"
    )

def build_group_followup(remaining: int, category: str, label: str) -> str:
    """
    Build a follow-up user message that asks to continue for the same subgroup.
    """
    group_phrase = category_constraint_phrase(category, label)
    return (
        f"Please continue with the remaining {remaining} question(s) "
        f"for {group_phrase}."
    )

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _single_run_generate_exact(
    service,
    model: str,
    base_message: str,
    exact_target: int,
    group_category: Optional[str] = None,
    group_label: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Generate up to 'exact_target' questions. If group constraints are provided,
    ensure the questions adhere to that group and try to hit the exact number.
    There is no hard round cap — we stop once the target is reached or progress stalls.
    Returns (text, actual_count).
    """
    run_id = str(uuid.uuid4())  # local trace id for this batch
    print(
        f"[generate:{run_id}] START exact_target={exact_target} model={model} "
        f"group=({group_category}={group_label})",
        flush=True,
    )

    # Build the initial message with or without the group constraint
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

    # Bootstrap the chat session
    session = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_first_message},
    ]

    class _UserMessageShim:
        """
        Minimal adapter to match service.get_response signature in providers.
        """
        def __init__(self, message: str, model: str):
            self.message = message
            self.model = model

    accumulated_text = ""

    # First round
    print(f"[generate:{run_id}] Calling service.get_response(first)...", flush=True)
    first_reply = service.get_response(_UserMessageShim(user_first_message, model), session)
    accumulated_text = first_reply or ""
    session.append({"role": "assistant", "content": first_reply})

    first_count = count_scenarios(accumulated_text)
    print(
        f"[generate:{run_id}] First reply length={len(accumulated_text)} "
        f"scenarios_found={first_count}",
        flush=True,
    )

    # If there are no XXX delimiters, we can't count reliably — return as-is
    if first_count == 0:
        print(f"[generate:{run_id}] No 'XXX' delimiters found → returning raw text.", flush=True)
        return accumulated_text, 0

    # Already hit or exceeded the target on the first round
    if first_count >= exact_target:
        if first_count > exact_target:
            print(
                f"[generate:{run_id}] Overshot ({first_count}>{exact_target}) → trimming.",
                flush=True,
            )
            accumulated_text = trim_to_n_questions(accumulated_text, exact_target)
        print(f"[generate:{run_id}] DONE (hit target in first round).", flush=True)
        return accumulated_text, min(first_count, exact_target)

    # Progress-driven loop (no hard cap)
    #
    # This loop asks the model for additional text until we reach the requested
    # `exact_target` number of scenarios or until progress stalls. Loop invariants
    # and break conditions:
    #  - `accumulated_text` always contains the concatenation of all assistant
    #    responses so far (and is used to count XXX-delimited scenarios).
    #  - We break when any of these is true:
    #      * current_count >= exact_target: we've reached the requested amount
    #      * assistant_text is empty/whitespace: model returned nothing useful
    #      * assistant_text contains no 'XXX' delimiters: no scenarios added
    #      * new_count <= last_count: no progress (prevent infinite loop)
    #
    # These checks ensure the loop is safe and deterministic even if the model
    # stops producing valid scenario blocks or returns malformed output.
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
        followup = (
            build_group_followup(remaining, group_category, group_label)
            if (group_category and group_label)
            else f"Please continue with the remaining {remaining} question(s)."
        )

        print(
            f"[generate:{run_id}] Asking follow-up (remaining={remaining}): {followup}",
            flush=True,
        )
        session.append({"role": "user", "content": followup})
        assistant_text = service.get_response(_UserMessageShim(followup, model), session)

        # If the model returned nothing (or whitespace), stop and accept partial
        if not assistant_text or assistant_text.strip() == "":
            print(f"[generate:{run_id}] Empty/whitespace reply → stop (partial).", flush=True)
            break

        accumulated_text += ("\n\n" if accumulated_text else "") + assistant_text
        session.append({"role": "assistant", "content": assistant_text})

        new_count = count_scenarios(accumulated_text)
        print(
            f"[generate:{run_id}] Chunk len={len(assistant_text)} "
            f"total_scenarios={new_count}",
            flush=True,
        )

        if new_count <= last_count:
            # The model didn't add any new XXX blocks; avoid infinite loop
            print(
                f"[generate:{run_id}] No progress (last={last_count}, new={new_count}) → break.",
                flush=True,
            )
            break
        last_count = new_count

    # Final safety trim if we overshot the requested count
    final_count = count_scenarios(accumulated_text)
    if final_count > exact_target:
        print(
            f"[generate:{run_id}] Final overshoot ({final_count}>{exact_target}) → trimming.",
            flush=True,
        )
        accumulated_text = trim_to_n_questions(accumulated_text, exact_target)
        final_count = exact_target

    print(f"[generate:{run_id}] END final_count={final_count}/{exact_target}", flush=True)
    return accumulated_text, final_count

def generate_texts(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate question generation for either:
    - a single batch (no demographics), or
    - multiple demographic batches (sum of per-group counts must equal target).

    Returns a structured dictionary with overall status, counts, final text,
    and detailed per-group metadata for transparency on the frontend.
    """
    message: str = payload.get("message", "")
    model: str = payload.get("model", "gpt-4o")
    print(payload)
    target: int = int(payload.get("number_of_questions", 1))
    demo: Optional[Dict[str, List[Dict[str, Any]]]] = payload.get("demographicData")

    if not message or not message.strip():
        raise ValueError("Payload 'message' must be a non-empty string.")
    
    session_id = str(uuid.uuid4())
    print(f"[generate] ===== NEW REQUEST =====", flush=True)
    print(f"[generate] session_id={session_id} model={model} target={target}", flush=True)

    service = select_service(model)

    chosen_category = pick_filled_category(demo)
    print(f"[generate:{session_id}] chosen_category={chosen_category}", flush=True)

    # -----------------------------------------------------------------------
    # CASE A: no demographic distribution → single-batch flow
    # -----------------------------------------------------------------------
    if not chosen_category:
        print(f"[generate:{session_id}] Case A (no demographic batching).", flush=True)
        print(
            f"[generate:{session_id}] BASE PROMPT:\n---8<---\n{message}\n---8<---",
            flush=True,
        )
        text, actual = _single_run_generate_exact(
            service=service,
            model=model,
            base_message=message,
            exact_target=target,
        )
        status = "completed" if actual >= target else "partial"
        print(f"[generate:{session_id}] status={status} actual={actual}/{target}", flush=True)

        if actual == 0:
            # No XXX blocks — return the raw text and explain why
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

    # -----------------------------------------------------------------------
    # CASE B: demographic distribution provided → multi-batch flow
    # -----------------------------------------------------------------------
    print(f"[generate:{session_id}] Case B (batched by {chosen_category}).", flush=True)
    groups = demo.get(chosen_category, []) if demo else []
    if not groups:
        print(f"[generate:{session_id}] WARNING: groups list empty → fallback to Case A.", flush=True)
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

    # Validate that the sum of requested per-group counts equals the global target
    total_requested = sum(int(g.get("value", 0)) for g in groups)
    print(f"[generate:{session_id}] groups={groups} total_requested={total_requested}", flush=True)
    if total_requested != target:
        raise ValueError(
            f"Demographic distribution total ({total_requested}) must equal number_of_questions ({target})."
        )

    combined_texts: List[str] = []
    combined_count = 0
    batches_meta: List[Dict[str, Any]] = []

    for idx, group in enumerate(groups, start=1):
        label: str = str(group["label"])
        qty: int = int(group["value"])
        print(
            f"[generate:{session_id}] ---- Group {idx}/{len(groups)}: "
            f"{chosen_category}='{label}', qty={qty}",
            flush=True,
        )

        # Log the effective prompt for this subgroup for full traceability
        eff_prompt = build_group_prompt(message, chosen_category, label, qty)
        print(
            f"[generate:{session_id}] Effective prompt for group:\n---8<---\n{eff_prompt}\n---8<---",
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

        # Collect rich metadata for the frontend (per-group transparency)
        batches_meta.append({
            "index": idx,
            "category": chosen_category,
            "label": label,
            "requested": qty,
            "actual": batch_actual,
            "prompt_used": eff_prompt.strip(),
            "text": batch_text,
        })

        # Add a human-readable header before each subgroup block in the combined text.
        # These headers do NOT affect scenario counting, since the regex matches only XXX...XXX blocks.
        header = f"### Group: {chosen_category}='{label}' ({batch_actual}/{qty})\n\n"
        combined_texts.append(header + (batch_text or ""))

        combined_count += batch_actual

    combined_response = "\n\n".join(t for t in combined_texts if t.strip())

    # Safety trim after concatenation (headers do not impact XXX block counting)
    total_after_concat = count_scenarios(combined_response)
    print(f"[generate:{session_id}] After concat scenarios={total_after_concat}", flush=True)
    if total_after_concat > target:
        print(f"[generate:{session_id}] Overshoot after concat → trimming to {target}.", flush=True)
        combined_response = trim_to_n_questions(combined_response, target)

    status = "completed" if combined_count >= target else "partial"
    print(f"[generate:{session_id}] DONE status={status} total_actual={combined_count}/{target}", flush=True)

    return {
        "status": status,
        "session_id": session_id,
        "model": model,
        "message": message,
        "requested_number_of_questions": target,
        "actual_number_of_questions": min(combined_count, target),
        "response": combined_response,
        "note": f"Batched by {chosen_category}.",
        "batches": batches_meta,  # Per-group metadata for the frontend
    }
