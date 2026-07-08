"""
Step C — Verdict synthesis. Deterministic rules, NOT another LLM call.
This is what makes the confidence score explainable.
"""

from src.utils.schemas import (
    ClaimVerdict,
    ExtractedClaim,
    GroundingResult,
    RegionVisibility,
    Verdict,
)

LOW_CERTAINTY_THRESHOLD = 0.5
QUALITY_PENALTY = 0.25


def synthesize_verdict(
    claim: ExtractedClaim,
    groundings: list[GroundingResult],
) -> ClaimVerdict:
    if not groundings:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            confidence=min(0.40, 0.0),
            supporting_grounding_ids=[],
        )

    visible_groundings = [
        g for g in groundings
        if g.region_visibility != RegionVisibility.NOT_VISIBLE
    ]

    if not visible_groundings:
        # Confidence here = claim substantiation level.
        # Evidence is absent so claim cannot be substantiated = low confidence.
        # We cap at 0.25 regardless of how certain we are the region is absent.
        return ClaimVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=Verdict.MISSING_EXPECTED_EVIDENCE,
            confidence=0.20,
            supporting_grounding_ids=[g.image_id for g in groundings],
        )

    # Contradiction check — highest priority finding
    contradicting = [g for g in visible_groundings if g.matches_claim is False]
    if contradicting:
        best = max(contradicting, key=lambda g: g.model_self_reported_certainty)
        conf = best.model_self_reported_certainty
        if best.image_quality_flag:
            conf = max(0.0, conf - QUALITY_PENALTY)
        return ClaimVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=Verdict.CONTRADICTED,
            confidence=conf,
            supporting_grounding_ids=[best.image_id],
        )

    # Supporting evidence found
    supporting = [g for g in visible_groundings if g.matches_claim is True]
    if supporting:
        best = max(supporting, key=lambda g: g.model_self_reported_certainty)
        conf = best.model_self_reported_certainty
        if best.image_quality_flag:
            conf = max(0.0, conf - QUALITY_PENALTY)

        # Partially visible → Partially Supported regardless of certainty
        if best.region_visibility == RegionVisibility.PARTIALLY_VISIBLE:
            verdict = Verdict.PARTIALLY_SUPPORTED
        # Low certainty even though visible → Partially Supported
        elif conf < LOW_CERTAINTY_THRESHOLD:
            verdict = Verdict.PARTIALLY_SUPPORTED
        else:
            verdict = Verdict.SUPPORTED

        return ClaimVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=verdict,
            confidence=conf,
            supporting_grounding_ids=[best.image_id],
        )

    # Visible but ambiguous
    return ClaimVerdict(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        confidence=min(0.40, min(g.model_self_reported_certainty for g in visible_groundings)),
        supporting_grounding_ids=[g.image_id for g in visible_groundings],
    )