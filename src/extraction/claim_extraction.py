"""
Step A — Claim extraction. Pure text reasoning, no images touched here.

Key fix: we give Gemini a minimal _ClaimsOnly schema instead of the full
ClaimExtractionResult. The model only fills in what it actually knows
(the claims themselves). We attach document_id and domain ourselves in
Python after validation — never ask the model to echo back values it
wasn't told.
"""

from src.utils.model_client import TextModelClient
from src.utils.schemas import ClaimExtractionResult, ExtractedClaim, Domain
from pydantic import BaseModel


# Internal schema — this is all Gemini ever sees
class _ClaimsOnly(BaseModel):
    claims: list[ExtractedClaim]


DOMAIN_HINTS = {
    Domain.LOAN_APPLICATION: (
        "Focus on claims about income, employment status, existing debts, collateral, "
        "and identity. Expected evidence is often a document photo/scan (payslip, bank "
        "statement, ID card) rather than a scene photo."
    ),
    Domain.VEHICLE_INSURANCE: (
        "Focus on claims about specific damaged vehicle parts and their severity. "
        "Each damaged part is a separate atomic claim. Expected evidence is a photo "
        "of the vehicle. Be precise about location: left/right, front/rear, which panel. "
        "Also extract claims about undamaged parts — these can be contradicted or confirmed."
    ),
    Domain.HEALTH_INSURANCE: (
        "Focus on claims about diagnosis, treatment received, injury location/severity, "
        "and dates of service. Expected evidence is often medical photos, scan reports, "
        "or itemized receipts."
    ),
    Domain.EVIDENCE_REVIEW: (
        "Focus on factual assertions that a piece of evidence is claimed to support. "
        "Expected evidence may be photos, scanned exhibits, or documents referenced by "
        "case claims."
    ),
    Domain.LICENSING_EMPLOYEE_VERIFICATION: (
        "Focus on claims about credential validity, issuing authority, expiry dates, "
        "role/title held, and identity match. Expected evidence is typically a photo of "
        "a license/ID card or badge."
    ),
}

EXTRACTION_PROMPT_TEMPLATE = """You are a compliance analyst. The text below may be a formal document OR
plain user-written description of a claim or loss. Either way, extract every
atomic, independently-checkable claim that could be verified by visual evidence.

Rules:
- Each claim must be atomic: one fact per claim.
- Assign short unique IDs: c1, c2, c3 etc.
- expected_region_or_field: be specific — "front left door panel",
  "front bumper surface", "roof tiles", not just "damage".
- claim_type: one of damage_description, financial_figure, identity_field,
  credential_validity, medical_condition, location_description.
- requires_visual_evidence: true for all claims you extract.
- Do not invent claims not present in the text.
- If two claims express the same underlying fact in different words, 
  keep only ONE — the more specific version.
- Never extract the same claim twice even if the document states it 
  multiple times or in multiple formats (narrative + table).
- After extracting, review your list and remove any claim that is 
  a restatement of another claim already in the list.

Domain: {domain}
Domain guidance: {domain_hint}

Input text:
---
{document_text}
---

Return JSON with a single key "claims" containing a list of claim objects.
"""

def extract_claims(
    client: TextModelClient,
    document_id: str,
    domain: Domain,
    document_text: str,
) -> ClaimExtractionResult:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        domain=domain.value,
        domain_hint=DOMAIN_HINTS[domain],
        document_text=document_text,
    )

    raw = client.structured_call(
        prompt_parts=[prompt],
        response_schema=_ClaimsOnly,
    )

    # Deduplicate — remove semantically redundant claims
    # Strategy: if two claims share more than 60% of their words, keep the longer one
    unique_claims = []
    for candidate in raw.claims:
        candidate_words = set(candidate.claim_text.lower().split())
        is_duplicate = False
        for existing in unique_claims:
            existing_words = set(existing.claim_text.lower().split())
            if len(candidate_words) == 0 or len(existing_words) == 0:
                continue
            overlap = len(candidate_words & existing_words)
            similarity = overlap / max(len(candidate_words), len(existing_words))
            if similarity > 0.60:
                # Keep the longer/more specific one
                if len(candidate.claim_text) > len(existing.claim_text):
                    unique_claims.remove(existing)
                    unique_claims.append(candidate)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_claims.append(candidate)

    # Re-assign clean sequential IDs after dedup
    for idx, claim in enumerate(unique_claims, 1):
        claim.claim_id = f"c{idx}"

    return ClaimExtractionResult(
        document_id=document_id,
        domain=domain,
        claims=unique_claims,
    )