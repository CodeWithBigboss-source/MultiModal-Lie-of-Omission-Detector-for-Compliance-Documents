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

from src.utils.model_client import ModelClient
from src.utils.schemas import (
    ExtractedClaim,
    GroundingResult,
    RegionVisibility,
)


# Internal schema — only what Gemini needs to fill in
# claim_id and image_id are attached by us in Python, not by the model
class _GroundingOnly(BaseModel):
    region_visibility: RegionVisibility
    description_of_what_is_seen: str
    matches_claim: Optional[bool]
    model_self_reported_certainty: float
    image_quality_flag: Optional[str]


GROUNDING_PROMPT_TEMPLATE = """You are examining a single photo/scan as evidence for one specific claim from a
compliance document. Answer ONLY about what is visible in THIS image.

Claim being checked: "{claim_text}"
Region/field that would need to be visible to check this claim: "{expected_region}"

Instructions:
- First determine whether that specific region/field is visible in the image at all.
  Set region_visibility to one of: fully_visible, partially_visible, not_visible.
- If not_visible, do not guess whether the claim is true or false.
  Set matches_claim to null. Absence of a region is NOT the same as contradiction.
- If visible, describe plainly what you see in description_of_what_is_seen, then
  set matches_claim to true if it matches the claim, false if it contradicts it,
  or null if ambiguous.
- Set model_self_reported_certainty between 0.0 and 1.0 honestly.
- Set image_quality_flag to blurry, low_resolution, poor_lighting if applicable,
  otherwise set it to null.

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
    client: ModelClient,
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