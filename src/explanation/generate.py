"""
Step D — Explanation phrasing only. Cannot change the verdict already decided
in Step C — enforced by only letting it output a single string field.
"""

from src.utils.model_client import ModelClient
from src.utils.schemas import ClaimVerdict, GroundingResult
from pydantic import BaseModel

EXPLANATION_PROMPT_TEMPLATE = """Write a single, clear sentence (compliance-report style, like an auditor's note)
explaining why this claim received this verdict. Do not restate the verdict label
itself as a separate word — weave it into the explanation naturally.

Claim: "{claim_text}"
Verdict: {verdict}
Confidence: {confidence:.2f}
Relevant grounding notes: {grounding_notes}

Return JSON matching the required schema exactly.
"""


class _ExplanationOnly(BaseModel):
    explanation: str


def generate_explanation(
    client: ModelClient,
    verdict: ClaimVerdict,
    groundings: list[GroundingResult],
) -> ClaimVerdict:
    relevant = [g for g in groundings if g.image_id in verdict.supporting_grounding_ids]
    notes = " | ".join(g.description_of_what_is_seen for g in relevant) or "No visible evidence in any submitted image."

    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        claim_text=verdict.claim_text,
        verdict=verdict.verdict.value,
        confidence=verdict.confidence,
        grounding_notes=notes,
    )
    result = client.structured_call(
        prompt_parts=[prompt],
        response_schema=_ExplanationOnly,
        prefer_lite=True,
    )
    verdict.explanation = result.explanation
    return verdict