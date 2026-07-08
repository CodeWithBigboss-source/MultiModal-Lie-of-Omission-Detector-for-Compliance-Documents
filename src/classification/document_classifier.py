"""
Document Classification — Phase 4b.
Reads extracted document text and automatically determines:
  - document_type: what kind of document this is
  - suggested_domain: which pipeline domain to route to
  - confidence: how certain the classifier is
  - reasoning: why it made this decision

Runs via Groq text model — fast, no local model needed.
"""

from pydantic import BaseModel
from src.utils.model_client import TextModelClient
from src.utils.schemas import Domain


class ClassificationResult(BaseModel):
    document_type: str
    suggested_domain: Domain
    confidence: float
    reasoning: str


CLASSIFICATION_PROMPT = """You are a document classifier for a compliance analysis system.

Read the document text below and determine:
1. What TYPE of document this is
2. Which compliance DOMAIN it belongs to
3. How CONFIDENT you are (0.0-1.0)
4. Brief REASONING for your decision

Document types you should recognize:
- court_ruling: judicial decisions, case verdicts, legal judgments
- medical_report: doctor notes, hospital records, diagnosis reports, lab results
- insurance_claim_form: insurance claim submissions, loss descriptions
- loan_application: credit applications, financial statements, income declarations
- payslip: salary slips, wage statements, employment income proof
- bank_statement: transaction records, account statements
- id_document: identity cards, passports, licenses
- vehicle_damage_report: accident reports, vehicle inspection reports
- property_inspection: site inspection reports, infrastructure assessments
- employment_contract: job offers, employment verification letters
- police_report: FIR, accident police reports
- receipt_invoice: purchase receipts, expense invoices
- general_claim: plain text claim description from user
- unknown: cannot determine

Domain mapping rules:
- vehicle_insurance: insurance_claim_form about vehicles, vehicle_damage_report, police_report about accidents
- health_insurance: medical_report, hospital records, pharmacy receipts
- loan_application: loan_application, payslip, bank_statement, employment_contract
- evidence_review: court_ruling, police_report, legal documents, case evidence
- licensing_employee_verification: id_document, employment_contract, professional licenses

Document text (first 1000 characters):
---
{text_preview}
---

Return JSON with fields: document_type, suggested_domain, confidence, reasoning.
suggested_domain must be exactly one of: loan_application, health_insurance, 
vehicle_insurance, evidence_review, licensing_employee_verification
"""


def classify_document(
    client: TextModelClient,
    extracted_text: str,
) -> ClassificationResult:
    """
    Classify a document from its extracted text.
    Returns suggested domain and document type.
    """
    # Only send first 1000 chars — enough to classify, saves tokens
    preview = extracted_text[:1000].strip()

    if not preview:
        return ClassificationResult(
            document_type="unknown",
            suggested_domain=Domain.EVIDENCE_REVIEW,
            confidence=0.0,
            reasoning="No text content found to classify."
        )

    prompt = CLASSIFICATION_PROMPT.format(text_preview=preview)

    try:
        result = client.structured_call(
            prompt=prompt,
            response_schema=ClassificationResult,
        )
        return result
    except Exception as e:
        return ClassificationResult(
            document_type="general_claim",
            suggested_domain=Domain.VEHICLE_INSURANCE,
            confidence=0.3,
            reasoning=f"Classification failed, defaulting to general claim: {e}"
        )