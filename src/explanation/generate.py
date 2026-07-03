"""
Step D — Explanation phrasing only. Cannot change the verdict already decided
in Step C — enforced by only letting it output a single string field.
"""

from src.utils.model_client import TextModelClient
from src.utils.schemas import ClaimVerdict, GroundingResult
from pydantic import BaseModel

class _ExplanationOnly(BaseModel):
    explanation: str

EXPLANATION_PROMPT_TEMPLATE = """Write a single clear sentence (compliance-report style, like an auditor's note)
explaining why this claim received this verdict. Do not restate the verdict label
as a separate word — weave it into the explanation naturally.

Claim: "{claim_text}"
Verdict: {verdict}
Confidence: {confidence:.2f}
Relevant grounding notes: {grounding_notes}

Return JSON with a single key "explanation" containing your sentence.
"""





def generate_explanation(
    client: TextModelClient,     # <-- changed from ModelClient
    verdict: ClaimVerdict,
    groundings: list[GroundingResult],
) -> ClaimVerdict:
    relevant = [g for g in groundings if g.image_id in verdict.supporting_grounding_ids]
    notes = (
        " | ".join(g.description_of_what_is_seen for g in relevant)
        or "No visible evidence in any submitted image."
    )
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        claim_text=verdict.claim_text,
        verdict=verdict.verdict.value,
        confidence=verdict.confidence,
        grounding_notes=notes,
    )
    result = client.structured_call(
        prompt=prompt,
        response_schema=_ExplanationOnly,
    )
    verdict.explanation = result.explanation
    return verdict