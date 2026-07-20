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


GROUNDING_PROMPT_TEMPLATE = """You are a forensic evidence reviewer. Study the image carefully.

CRITICAL ORIENTATION RULE:
Describe ALL positions using CAMERA FRAME POSITION only.
- Use: "left side of frame", "right side of frame", "center of frame", "top of frame"
- NEVER use: "driver side", "passenger side", "vehicle's left", "vehicle's right"
- The camera angle determines what "left" and "right" mean — not the vehicle's orientation
- What appears on the RIGHT side of the photo IS the right side. State it as such.

STEP 1 — MAP THE CAMERA FRAME:
Before reading the claim, answer:
- What is the camera angle? (front-left, right side, rear, etc. using frame position)
- Which vehicle components are PHYSICALLY VISIBLE within this frame?
- What components are NOT visible / outside the frame?
- What damage is visible on each component? Be specific.

STEP 2 — EVALUATE THE CLAIM:
Claim: "{claim_text}"
Expected region: "{expected_region}"

Using ONLY your Step 1 frame map:
- Is the expected region within the camera frame?
  fully_visible / partially_visible / not_visible
- If not_visible: matches_claim MUST be null. Do not guess.
- If visible and confirms claim: matches_claim = true
- If visible and contradicts claim: matches_claim = false
- If visible but ambiguous: matches_claim = null

HARD RULES:
1. Do NOT need full vehicle in frame. If the claimed part is visible, that is enough.
2. NEVER infer condition of parts outside the frame.
3. If claim says damage exists and you clearly see that damage: matches_claim = true
4. Region clearly absent from frame: certainty 0.85-0.95
5. Region visible and condition clearly matches: certainty 0.85-0.95

description_of_what_is_seen: Write your complete Step 1 frame map here.
Use camera frame positions throughout (left of frame, right of frame, etc.)
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