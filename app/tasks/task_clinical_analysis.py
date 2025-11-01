"""Clinical scenario analysis orchestration and provider selection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.services.einfra_service import EInfraService
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService

# import extracted helpers
from app.utils.analysis_helpers import extract_scenarios, chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def select_service(model_name: str):
    """Select a provider service class based on the model prefix."""
    if model_name.startswith("llama"):
        print(
            f"[analyze] select_service → EInfraService (model='{model_name}')",
            flush=True,
        )
        return EInfraService()
    if model_name.startswith("gemini"):
        print(
            f"[analyze] select_service → GeminiService (model='{model_name}')",
            flush=True,
        )
        return GeminiService()
    if model_name.startswith("gpt"):
        print(
            f"[analyze] select_service → OpenAIService (model='{model_name}')",
            flush=True,
        )
        return OpenAIService()
    raise ValueError(
        f"Invalid model specified: {model_name}. Supported: llama*, gemini*, gpt-*."
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 3  # Process at most three scenarios per LLM call


def analyze_clinical_questions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate clinical scenario extraction and normalization.

    Steps:
      1. Parse scenarios from raw input text (delimited by ``XXX ... XXX``).
      2. Batch scenarios (default: 3 per batch).
      3. Call the selected provider to extract structured fields.
      4. Normalize and align results with inputs.
      5. Merge batches and return a unified structure.

    Returns:
        {'questions': [{ 'gender': ..., 'ethnicity': ..., 'age': ...}, ...]}
    """
    model: str = payload.get("model", "gpt-4o")
    questions: str = payload.get("message", "")
    service = select_service(model)

    # Step 1: Extract delimited scenarios in order.
    indexed_scenarios = extract_scenarios(questions)
    print(f"[analyze] model={model}", flush=True)
    print(f"[analyze] payload keys={list(payload.keys())}", flush=True)
    print(f"[analyze] extracted {len(indexed_scenarios)} scenario(s)", flush=True)
    if indexed_scenarios:
        print(
            f"[analyze] first scenario preview: {indexed_scenarios[0][1][:120]!r}",
            flush=True,
        )
    if not indexed_scenarios:
        # No delimited scenarios found → return empty structure.
        return {"questions": []}

    # Collect results as (original_index, extracted_object)
    collected: List[Tuple[int, Dict[str, Any]]] = []

    # Step 2: Process scenarios in batches.
    for batch in chunk(indexed_scenarios, DEFAULT_BATCH_SIZE):
        # Combine batch scenarios into a single text block.
        batch_text = "\n\n".join(s for (_idx, s) in batch)
        batch_count = len(batch)

        # Step 3: Provider extraction call.
        extracted = service.extract_clinical_info(
            questions=batch_text,
            number_of_questions=batch_count,
        )

        # Step 4: Normalize provider response to expected shape.
        # Expect a dict with key "questions" → list[dict].
        if not isinstance(extracted, dict) or "questions" not in extracted:
            if isinstance(extracted, list):
                extracted_questions = extracted
            else:
                extracted_questions = [
                    {"gender": None, "ethnicity": None, "age": None}
                ] * batch_count
        else:
            extracted_questions = extracted.get("questions", [])

        # Step 5: Enforce 1:1 mapping (pad or truncate as needed).
        if len(extracted_questions) < batch_count:
            deficit = batch_count - len(extracted_questions)
            extracted_questions += [
                {"gender": None, "ethnicity": None, "age": None}
            ] * deficit
        elif len(extracted_questions) > batch_count:
            extracted_questions = extracted_questions[:batch_count]

        # Step 6: Attach to original indices.
        for (orig_idx, _), obj in zip(batch, extracted_questions):
            collected.append((orig_idx, obj))

    # Step 7: Sort by original index and merge.
    collected.sort(key=lambda x: x[0])
    merged_results = [r for (_i, r) in collected]

    # Step 8: Final structure for the frontend.
    return {"questions": merged_results}
