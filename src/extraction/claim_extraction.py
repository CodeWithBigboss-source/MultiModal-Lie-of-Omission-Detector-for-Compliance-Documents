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

EXTRACTION_PROMPT_TEMPLATE = """You are a compliance analyst. Read the document text below and extract every
atomic, independently-checkable claim that COULD be verified or contradicted by a
piece of visual evidence (photo, scanned document, ID card, etc).

Rules:
- Each claim must be atomic: one fact per claim, not a compound sentence.
- For each claim, state exactly what visual region/object/field a reviewer would
  need to SEE in order to check it.
- Assign each claim a short unique claim_id like "c1", "c2", "c3".
- For claim_type use one of: damage_description, financial_figure, identity_field,
  credential_validity, medical_condition, location_description.
- Skip claims that are pure opinion or that no image could ever verify.
- Do not invent claims not present in the text.
- requires_visual_evidence should be true for all claims you extract.

Domain: {domain}
Domain-specific guidance: {domain_hint}

Document text:
---
{document_text}
---

Return a JSON object with a single key "claims" containing a list of claim objects.
"""


def extract_claims(
    client: TextModelClient,     # <-- changed from ModelClient
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
        prompt=prompt,             # <-- Groq takes a single string, not a list
        response_schema=_ClaimsOnly,
    )
    return ClaimExtractionResult(
        document_id=document_id,
        domain=domain,
        claims=raw.claims,
    )