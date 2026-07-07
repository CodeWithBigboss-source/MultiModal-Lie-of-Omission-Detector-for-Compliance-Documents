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
    region_visibility: RegionVisibility
    description_of_what_is_seen: str
    matches_claim: Optional[bool]
    model_self_reported_certainty: float
    image_quality_flag: Optional[str]


GROUNDING_PROMPT_TEMPLATE = """You are a strict forensic evidence reviewer. Your ONLY job is to report 
what is GEOMETRICALLY PRESENT within the camera frame. 

ABSOLUTE RULES — violating these is a critical failure:
- If a region is NOT within the camera frame boundary, it is NOT_VISIBLE. Period.
- NEVER infer, assume, or guess about regions outside the frame.
- NEVER say a region is visible because you expect it to be there.
- A vehicle photo showing the LEFT side tells you NOTHING about the RIGHT side.
- A vehicle photo showing the FRONT tells you NOTHING about the REAR.
- When in doubt about visibility → not_visible.

STEP 1 — MAP THE FRAME (do this before reading the claim):
Answer these questions strictly about what is inside the camera frame:
- Which side of the vehicle is shown? (left/right/front/rear/front-left angle etc.)
- Which specific panels/components are clearly within the frame?
- What damage is visible on those specific panels?
- What panels are partially at the edge of the frame?
- What panels/regions are completely outside the frame?

STEP 2 — EVALUATE THE CLAIM:
Claim: "{claim_text}"
Expected region: "{expected_region}"

Using ONLY your Step 1 observations:
- Is the expected region inside the camera frame?
  fully_visible / partially_visible / not_visible
- If not_visible: matches_claim MUST be null. Do not guess.
- If the claimed region IS visible and what you see confirms the claim → 
  matches_claim = true. You do NOT need to see the full vehicle. 
  You only need to see the specific claimed region. If the left door 
  is visible and damaged, that is sufficient to confirm a claim about 
  the left door — period.
- If visible and evidence contradicts claim: matches_claim = false
- If visible but unclear: matches_claim = null

STEP 3 — CONFIDENCE:
- Be honest. If you mapped the frame carefully and region is clearly absent → 
  model_self_reported_certainty should be HIGH (0.85-0.95).
- If region is clearly visible and damage clearly matches → HIGH certainty.
- A claim about one specific part being confirmed by that specific part 
  being clearly visible and damaged = HIGH certainty. Do not penalize 
  for not seeing other parts of the vehicle.
  
description_of_what_is_seen: Write your complete Step 1 frame map here.
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