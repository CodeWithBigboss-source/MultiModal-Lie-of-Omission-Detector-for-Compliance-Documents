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


GROUNDING_PROMPT_TEMPLATE = """You are a strict compliance evidence reviewer. 

PHASE 1 — Describe ALL visible evidence in this image comprehensively and objectively.
List every damaged area, every visible feature, every relevant detail you can see.
Do NOT skip anything visible. Do NOT assume anything not visible.

PHASE 2 — Now evaluate this specific claim against what you described in Phase 1:
Claim: "{claim_text}"
Expected region/field to verify: "{expected_region}"

Strict rules:
- region_visibility must reflect reality: if that specific region is NOT clearly
  in the frame, set not_visible — even if other damage is present.
- matches_claim: true only if the evidence directly confirms the claim.
  false if it contradicts it. null if the region is absent or ambiguous.
- description_of_what_is_seen: write your FULL Phase 1 observation here —
  describe everything visible in the image, not just the claimed region.
  This is the most important field — be specific and factual.
- model_self_reported_certainty: be honest. If you cannot clearly see the
  region, lower your certainty. Never hallucinate visibility.
- image_quality_flag: blurry / low_resolution / poor_lighting or null.

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