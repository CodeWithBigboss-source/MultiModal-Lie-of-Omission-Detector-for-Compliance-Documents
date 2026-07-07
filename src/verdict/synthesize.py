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
            confidence=0.0,
            supporting_grounding_ids=[],
        )

    visible_groundings = [
        g for g in groundings
        if g.region_visibility != RegionVisibility.NOT_VISIBLE
    ]

    if not visible_groundings:
        # Region not visible in ANY image — this is the lie-of-omission case.
        # We are CONFIDENT it is missing, so confidence should be HIGH.
        best_certainty = max(g.model_self_reported_certainty for g in groundings)
        return ClaimVerdict(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=Verdict.MISSING_EXPECTED_EVIDENCE,
            confidence=max(best_certainty, 0.85),  # floor at 0.85 — we're sure it's not there
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
        confidence=min(g.model_self_reported_certainty for g in visible_groundings),
        supporting_grounding_ids=[g.image_id for g in visible_groundings],
    )