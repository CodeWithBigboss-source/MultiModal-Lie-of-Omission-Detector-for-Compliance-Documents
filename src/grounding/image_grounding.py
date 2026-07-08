"""
Step B — Evidence grounding.
One narrow question per (claim, image) pair.
Local OpenCV quality check runs first to save API calls on unusable images.
"""

from typing import Optional
from PIL import Image
import cv2
import numpy as np
from pydantic import BaseModel

from src.utils.model_client import VisionModelClient
from src.utils.schemas import (
    ExtractedClaim,
    GroundingResult,
    RegionVisibility,
)


# Internal schema — only what Gemini needs to fill in
# claim_id and image_id are attached by us in Python, not by the model
class _GroundingOnly(BaseModel):
    model_config = {"protected_namespaces": ()}

    region_visibility: RegionVisibility
    description_of_what_is_seen: str
    matches_claim: Optional[bool]
    model_self_reported_certainty: float
    image_quality_flag: Optional[str]


GROUNDING_PROMPT_TEMPLATE = """You are a forensic evidence reviewer. Study the image carefully before reading anything below.

═══ STEP 1 — IDENTIFY CAMERA ANGLE AND FRAME BOUNDARIES ═══
Before anything else, answer:
- What is the primary camera angle? (e.g. front-left, left side, rear-right, etc.)
- List ONLY the vehicle components that are PHYSICALLY WITHIN this camera frame.
- Explicitly state what is NOT visible: "The right side is not visible. The rear is not visible."

This step is mandatory. Do not skip it.

═══ STEP 2 — EVALUATE THE CLAIM ═══
Claim: "{claim_text}"
Region to verify: "{expected_region}"

Using ONLY your Step 1 list of visible components:

HARD RULES — these override everything else:
- If the expected region was NOT in your Step 1 visible list → region_visibility = not_visible
- If region_visibility = not_visible → matches_claim MUST be null. No exceptions. Ever.
- You CANNOT infer the condition of a region that is outside the camera frame.
- Seeing one side of a vehicle tells you NOTHING about the opposite side.
- Do not write phrases like "part of X is visible at the edge" unless it is literally at the pixel edge of the frame.

IF REGION IS VISIBLE:
- Describe exactly what you see in that specific region
- matches_claim = true if condition matches the claim
- matches_claim = false if condition directly contradicts the claim
- matches_claim = null if genuinely ambiguous

CONFIDENCE RULES:
- Region clearly absent from frame → certainty 0.90 (you are sure it is not there)
- Region clearly visible and condition clearly matches → certainty 0.85-0.95
- Region partially visible or condition ambiguous → certainty 0.50-0.70
- Never inflate confidence when uncertain

description_of_what_is_seen: Write your Step 1 frame analysis here — 
list what IS visible and explicitly state what is NOT visible.
image_quality_flag: blurry / low_resolution / poor_lighting / null

Return JSON matching the required schema exactly.
"""


def _quality_flag(image: Image.Image) -> str | None:
    """Cheap local OpenCV heuristic — no API call needed."""
    arr = np.array(image.convert("L"))
    laplacian_var = cv2.Laplacian(arr, cv2.CV_64F).var()
    h, w = arr.shape
    if min(h, w) < 400:
        return "low_resolution"
    if laplacian_var < 50:
        return "blurry"
    return None


def ground_claim_against_image(
    client: VisionModelClient,
    claim: ExtractedClaim,
    image: Image.Image,
    image_id: str,
) -> GroundingResult:
    quality_flag = _quality_flag(image)

    prompt = GROUNDING_PROMPT_TEMPLATE.format(
        claim_text=claim.claim_text,
        expected_region=claim.expected_region_or_field,
    )

    raw = client.structured_call(
        prompt_parts=[prompt, image],
        response_schema=_GroundingOnly,
    )

    return GroundingResult(
        claim_id=claim.claim_id,
        image_id=image_id,
        region_visibility=raw.region_visibility,
        description_of_what_is_seen=raw.description_of_what_is_seen,
        matches_claim=raw.matches_claim,
        model_self_reported_certainty=raw.model_self_reported_certainty,
        image_quality_flag=quality_flag or raw.image_quality_flag,
    )