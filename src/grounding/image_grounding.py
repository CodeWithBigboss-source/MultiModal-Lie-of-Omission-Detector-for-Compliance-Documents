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


GROUNDING_PROMPT_TEMPLATE = """You are a forensic evidence reviewer. Your job is to evaluate ONE specific claim against ONE image.

═══ STEP 1 — FRAME MAPPING (read the image before reading the claim) ═══
Precisely list:
a) Camera angle: which side of the subject is shown? (front-left, left side only, rear, etc.)
b) Every component clearly within the frame and its condition
c) Every component partially at the frame edge
d) What is completely outside the frame — do not infer its condition

═══ STEP 2 — CLAIM EVALUATION ═══
Claim: "{claim_text}"
Region to verify: "{expected_region}"

Answer strictly from Step 1 only:

VISIBILITY:
- Is the expected region within the camera frame?
- fully_visible = clearly in frame with enough detail to assess
- partially_visible = at the edge, cut off, or obscured
- not_visible = outside the frame or completely hidden

MATCH:
- If not_visible → matches_claim MUST be null. No exceptions.
- If fully or partially visible AND condition confirms claim → true
- If fully or partially visible AND condition contradicts claim → false
- If visible but genuinely unclear → null

CRITICAL RULES:
1. You do NOT need the full subject in frame. If the claimed part is visible, that is enough.
2. Never assume condition of parts outside the frame.
3. If you see damage that clearly contradicts "minor scratches" or "no damage" → false immediately.
4. High confidence when region is clearly absent from frame (0.85-0.95).
5. High confidence when damage clearly matches or contradicts claim (0.85-0.95).
6. Low confidence only for genuinely ambiguous image quality or partial occlusion.

description_of_what_is_seen: your complete Step 1 frame map goes here.
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