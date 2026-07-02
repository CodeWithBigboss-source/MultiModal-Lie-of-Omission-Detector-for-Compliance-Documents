"""
Step B — Evidence grounding. One narrow question per (claim, image) pair.
Local OpenCV quality check runs first to save API calls on unusable images.
"""

from PIL import Image
import cv2
import numpy as np

from src.utils.model_client import ModelClient
from src.utils.schemas import ExtractedClaim, GroundingResult

GROUNDING_PROMPT_TEMPLATE = """You are examining a single photo/scan as evidence for one specific claim from a
compliance document. Answer ONLY about what is visible in THIS image.

Claim being checked: "{claim_text}"
Region/field that would need to be visible to check this claim: "{expected_region}"

Instructions:
- First determine whether that specific region/field is visible in the image at all:
  fully_visible, partially_visible, or not_visible.
- If not_visible, do not guess whether the claim is true or false — say so explicitly
  and set matches_claim to null. Absence of the region is NOT the same as contradiction.
- If visible, describe plainly what you see, then state whether it matches the claim
  (true), contradicts it (false), or is ambiguous (null).
- Report your own certainty honestly (0.0-1.0).

Return JSON matching the required schema exactly.
"""


def _quality_flag(image: Image.Image) -> str | None:
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
    result = client.structured_call(
        prompt_parts=[prompt, image],
        response_schema=GroundingResult,
    )
    result.claim_id = claim.claim_id
    result.image_id = image_id
    result.image_quality_flag = quality_flag or result.image_quality_flag
    return result