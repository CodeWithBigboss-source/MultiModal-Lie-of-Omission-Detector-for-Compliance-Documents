"""
Step A — Claim extraction. Pure text reasoning, no images touched here.
"""

from src.utils.model_client import ModelClient
from src.utils.schemas import ClaimExtractionResult, Domain

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
- For each claim, state exactly what visual region/object/field a reviewer would need
  to SEE in order to check it.
- Skip claims that are pure opinion or that no image could ever verify.
- Do not invent claims that aren't in the text.

Domain: {domain}
Domain-specific guidance: {domain_hint}

Document text:
---
{document_text}
---

Return JSON matching the required schema exactly.
"""


def extract_claims(
    client: ModelClient,
    document_id: str,
    domain: Domain,
    document_text: str,
) -> ClaimExtractionResult:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        domain=domain.value,
        domain_hint=DOMAIN_HINTS[domain],
        document_text=document_text,
    )
    result = client.structured_call(
        prompt_parts=[prompt],
        response_schema=ClaimExtractionResult,
    )
    result.document_id = document_id
    result.domain = domain
    return result