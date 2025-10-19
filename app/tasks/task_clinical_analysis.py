import re
from typing import Any, Dict, List, Tuple
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Regex pattern to find clinical scenarios delimited by XXX...XXX,
# allowing matches across multiple lines and ignoring case.
SCENARIO_PATTERN = re.compile(r'XXX([\s\S]*?)XXX', re.IGNORECASE)

def extract_scenarios(text: str) -> List[Tuple[int, str]]:
    """
    Extract all scenarios delimited by XXX...XXX.
    Returns a list of tuples (original_index, scenario_text).
    The index preserves the original order for later reassembly.
    """
    scenarios = []
    for i, m in enumerate(SCENARIO_PATTERN.finditer(text)):
        scenario = m.group(1).strip()  # Extract the content inside the delimiters
        if scenario:
            scenarios.append((i, scenario))
    return scenarios


def chunk(lst, size: int):
    """
    Yield successive chunks of a list, each of a specified maximum size.
    Used to process long inputs in smaller groups.
    """
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def select_service(m: str):
    """
    Pick the model provider (Gemini or OpenAI) based on the model name prefix.
    """
    if m.startswith("gemini"):
        print(f"[generate] select_service → GeminiService (model='{m}')", flush=True)
        return GeminiService()
    if m.startswith("gpt"):
        print(f"[generate] select_service → OpenAIService (model='{m}')", flush=True)
        return OpenAIService()
    raise ValueError(f"Invalid model specified: {m}. Supported: gemini*, gpt-*.")


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 3  # Process at most three scenarios per LLM call


def analyze_clinical_questions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate clinical scenario analysis by:
    - parsing scenarios from the raw input text (delimited by XXX...XXX),
    - batching them in small groups (default: 3 per batch),
    - invoking the selected model provider for structured data extraction,
    - aggregating all batch results while preserving the original order.

    Returns a structured dictionary aligned with JobFinishedResponse,
    containing all extracted demographic information for each scenario.
    """
  
    model: str = payload.get("model", "gpt-4o")
    questions: str = payload.get("message", "")
    service = select_service(model)

    
    
    # Step 1: Extract all scenarios from the raw text.
    # Each element is a tuple (index, scenario_text).
    indexed_scenarios = extract_scenarios(questions)
    print(f"[analyze] model={model}", flush=True)
    print(f"[analyze] payload keys={list(payload.keys())}", flush=True)
    print(f"[analyze] extracted {len(indexed_scenarios)} scenario(s)", flush=True)
    if indexed_scenarios:
        print(f"[analyze] first scenario preview: {indexed_scenarios[0][1][:120]!r}", flush=True)
    if not indexed_scenarios:
        # If no delimited scenarios are found, return an empty result.
        return {"questions": []}

    
    # List to collect results as (original_index, extracted_info)
    collected: List[Tuple[int, dict]] = []

    # Step 2: Process scenarios in batches of fixed size (3 by default).
    for batch in chunk(indexed_scenarios, DEFAULT_BATCH_SIZE):
        # Combine the scenarios in this batch into a single text block.
        # Separating them with blank lines helps the model distinguish them.
        batch_text = "\n\n".join([s for (_idx, s) in batch])
        batch_count = len(batch)

        # Step 3: Call the model provider to extract structured information.
        extracted = service.extract_clinical_info(
            questions=batch_text,
            number_of_questions=batch_count,
        )

        # Step 4: Normalize the provider’s response to ensure a consistent shape.
        # Expecting a dict with a "questions" key containing a list of results.
        if not isinstance(extracted, dict) or "questions" not in extracted:
            # If an unexpected structure is returned, adapt or fill with placeholders.
            if isinstance(extracted, list):
                extracted_questions = extracted
            else:
                extracted_questions = [
                    {"gender": None, "ethnicity": None, "age": None}
                ] * batch_count
        else:
            extracted_questions = extracted.get("questions", [])

        # Step 5: Enforce alignment between number of inputs and outputs.
        # Pad with empty objects or truncate to maintain one-to-one mapping.
        if len(extracted_questions) < batch_count:
            deficit = batch_count - len(extracted_questions)
            extracted_questions += [
                {"gender": None, "ethnicity": None, "age": None}
            ] * deficit
        elif len(extracted_questions) > batch_count:
            extracted_questions = extracted_questions[:batch_count]

        # Step 6: Attach results to their original indices for proper ordering.
        for (orig_idx, _s), result_obj in zip(batch, extracted_questions):
            collected.append((orig_idx, result_obj))

    # Step 7: Sort results by their original index and merge into a single list.
    collected.sort(key=lambda x: x[0])
    merged_results = [r for (_i, r) in collected]

    # Step 8: Return the final structured response expected by the frontend.
    return {"questions": merged_results}
