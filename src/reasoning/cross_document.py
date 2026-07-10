"""
Cross-Document Contradiction Detection — Phase 5.

When multiple documents are uploaded, this module:
1. Extracts claims from each document separately with its source label
2. Compares claims across documents to find factual contradictions
3. Flags where Document A says X but Document B says Y
4. Distinguishes factual contradictions from legal disagreements

Example: Court ruling says "no damage was observed at the scene"
         Police report says "severe front-end collision damage noted"
         → CROSS-DOCUMENT CONTRADICTION flagged
"""

from pydantic import BaseModel
from typing import Optional
from src.utils.model_client import TextModelClient


class CrossDocumentClaim(BaseModel):
    claim_id: str
    source_document: str
    claim_text: str
    claim_type: str


class Contradiction(BaseModel):
    contradiction_id: str
    claim_a: str          # claim text from document A
    source_a: str         # document A label
    claim_b: str          # claim text from document B
    source_b: str         # document B label
    contradiction_type: str  # factual / temporal / identity / severity
    explanation: str
    severity: str         # high / medium / low


class CrossDocumentReport(BaseModel):
    total_documents: int
    total_claims_extracted: int
    contradictions: list[Contradiction]
    consistent_claims: list[str]
    overall_assessment: str


CROSS_DOC_PROMPT = """You are a forensic document analyst performing cross-document 
consistency analysis for a compliance review.

You have been given claims extracted from multiple documents. Your job is to:
1. Identify factual contradictions between claims from DIFFERENT documents
2. Classify the type of contradiction
3. Assess severity
4. Distinguish between factual errors and mere differences in perspective

IMPORTANT BOUNDARIES:
- Only flag FACTUAL contradictions — where one document states a fact that 
  directly contradicts a fact in another document
- Do NOT flag legal interpretation differences as contradictions
- Do NOT flag contradictions within the same document
- A court ruling saying "insufficient evidence" does NOT contradict a report 
  saying "damage was present" — legal sufficiency is not a factual contradiction

Contradiction types:
- factual: direct factual disagreement (damage present vs no damage)
- temporal: date/time disagreements 
- identity: person/vehicle/location identity conflicts
- severity: same event described with very different severity levels
- presence: one doc says X was present, another says X was absent

Severity levels:
- high: directly affects claim outcome
- medium: relevant but may have innocent explanation
- low: minor discrepancy, likely clerical

Documents and their claims:
---
{documents_and_claims}
---

Return JSON matching the required schema exactly.
"""


_ClaimsFromDocPrompt = """Extract all atomic, independently-verifiable factual 
claims from this document. Each claim should be one specific fact.

Document label: {doc_label}
Document text:
---
{doc_text}
---

For each claim assign:
- claim_id: "{doc_label}_c1", "{doc_label}_c2" etc.
- source_document: "{doc_label}"
- claim_text: the atomic factual claim
- claim_type: one of: damage_description, financial_figure, identity_field, 
  temporal_fact, location_fact, medical_condition, credential_validity, 
  legal_finding, procedural_fact

Return JSON with key "claims" containing list of claim objects.
"""


class _ClaimsOnly(BaseModel):
    claims: list[CrossDocumentClaim]


def extract_claims_from_document(
    client: TextModelClient,
    doc_label: str,
    doc_text: str,
) -> list[CrossDocumentClaim]:
    prompt = _ClaimsFromDocPrompt.format(
        doc_label=doc_label,
        doc_text=doc_text[:3000],
    )
    try:
        result = client.structured_call(
            prompt=prompt,
            response_schema=_ClaimsOnly,
        )
        return result.claims
    except Exception as e:
        print(f"Claim extraction failed for {doc_label}: {e}")
        return []


def detect_cross_document_contradictions(
    client: TextModelClient,
    documents: dict[str, str],  # label -> extracted text
) -> CrossDocumentReport:
    """
    documents: dict mapping document label to extracted text
    e.g. {"court_ruling": "...", "police_report": "...", "insurance_form": "..."}
    """
    if len(documents) < 2:
        return CrossDocumentReport(
            total_documents=len(documents),
            total_claims_extracted=0,
            contradictions=[],
            consistent_claims=[],
            overall_assessment=(
                "At least two documents are required for "
                "cross-document contradiction analysis."
            )
        )

    # Extract claims from each document
    all_claims: list[CrossDocumentClaim] = []
    for label, text in documents.items():
        claims = extract_claims_from_document(client, label, text)
        all_claims.extend(claims)

    # Format claims for contradiction detection prompt
    doc_sections = []
    for label in documents.keys():
        doc_claims = [c for c in all_claims if c.source_document == label]
        claims_text = "\n".join(
            f"  [{c.claim_id}] {c.claim_text}"
            for c in doc_claims
        )
        doc_sections.append(f"DOCUMENT: {label}\n{claims_text}")

    documents_and_claims = "\n\n".join(doc_sections)

    prompt = CROSS_DOC_PROMPT.format(
        documents_and_claims=documents_and_claims
    )

    try:
        result = client.structured_call(
            prompt=prompt,
            response_schema=CrossDocumentReport,
        )
        result.total_documents = len(documents)
        result.total_claims_extracted = len(all_claims)
        return result
    except Exception as e:
        return CrossDocumentReport(
            total_documents=len(documents),
            total_claims_extracted=len(all_claims),
            contradictions=[],
            consistent_claims=[],
            overall_assessment=f"Cross-document analysis failed: {e}"
        )