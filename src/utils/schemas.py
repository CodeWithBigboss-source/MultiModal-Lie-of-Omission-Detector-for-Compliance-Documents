"""
Core data contracts for the pipeline.

Every module (extraction -> grounding -> verdict -> explanation) reads and
writes these exact shapes. Keeping this in one file means when we add a new
domain (loan / health insurance / evidence review / licensing), we only
touch the domain config, never these schemas.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domains we've scoped for this project
# ---------------------------------------------------------------------------
class Domain(str, Enum):
    LOAN_APPLICATION = "loan_application"
    VEHICLE_INSURANCE = "vehicle_insurance"
    HEALTH_INSURANCE = "health_insurance"
    EVIDENCE_REVIEW = "evidence_review"
    LICENSING_EMPLOYEE_VERIFICATION = "licensing_employee_verification"


# ---------------------------------------------------------------------------
# Step A output: one atomic, checkable claim extracted from the document
# ---------------------------------------------------------------------------
class ExtractedClaim(BaseModel):
    claim_id: str
    claim_text: str = Field(..., description="The atomic claim, in the document's own words or a faithful paraphrase")
    expected_region_or_field: str = Field(
        ..., description="What visual region/object/field would need to be visible to check this claim, "
                          "e.g. 'left rear door of vehicle', 'signature block', 'expiry date on license card'"
    )
    claim_type: str = Field(..., description="e.g. 'damage_description', 'financial_figure', 'identity_field', 'credential_validity'")
    requires_visual_evidence: bool = Field(..., description="Whether checking this claim requires visual evidence")

class ClaimExtractionResult(BaseModel):
    document_id: str
    domain: Domain
    claims: list[ExtractedClaim]


# ---------------------------------------------------------------------------
# Step B output: grounding result for ONE claim against ONE image
# ---------------------------------------------------------------------------
class RegionVisibility(str, Enum):
    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"


class GroundingResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    claim_id: str
    image_id: str
    region_visibility: RegionVisibility
    description_of_what_is_seen: str
    matches_claim: Optional[bool] = Field(
    ..., description="True/False only meaningful if region_visibility != NOT_VISIBLE. "
                      "Explicitly output null if it cannot be determined."
    )
    model_self_reported_certainty: float = Field(..., ge=0.0, le=1.0)
    image_quality_flag: Optional[str] = Field(..., description="e.g. 'blurry', 'low_resolution', 'poor_lighting'. Explicitly output null if no issue.")


# ---------------------------------------------------------------------------
# Step C output: final verdict per claim
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"
    MISSING_EXPECTED_EVIDENCE = "Missing Expected Evidence"


class ClaimVerdict(BaseModel):
    claim_id: str
    claim_text: str
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_grounding_ids: list[str]
    explanation: Optional[str] = None  # filled in by Step D


# ---------------------------------------------------------------------------
# Final report for one document
# ---------------------------------------------------------------------------
class ComplianceReport(BaseModel):
    document_id: str
    domain: Domain
    claim_verdicts: list[ClaimVerdict]
    overall_risk_note: Optional[str] = None