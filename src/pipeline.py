"""
End-to-end orchestrator with PII layer.
1. PII strips text and masks images locally before any API call.
2. Pipeline runs on anonymized data.
3. PII registry restores real values in explanations before returning.
"""

from PIL import Image

from src.extraction.claim_extraction import extract_claims
from src.grounding.image_grounding import ground_claim_against_image
from src.verdict.synthesize import synthesize_verdict
from src.explanation.generate import generate_explanation
from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import ComplianceReport, Domain, ClaimVerdict, Verdict
from src.pii.detector import PIIRegistry, mask_text, mask_image


def run_pipeline(
    text_client: TextModelClient,
    vision_client: VisionModelClient,
    document_id: str,
    domain: Domain,
    document_text: str,
    images: dict[str, Image.Image],
) -> ComplianceReport:

    # ── PII Layer ────────────────────────────────────────────────────────────
    registry = PIIRegistry()

    # Mask sensitive text before it goes to any API
    masked_text = mask_text(document_text, registry)

    # Mask faces and plates in every image before they go to any API
    masked_images = {}
    image_blur_notes = {}
    for img_id, img in images.items():
        masked_img, blurred = mask_image(img)
        masked_images[img_id] = masked_img
        if blurred:
            image_blur_notes[img_id] = blurred

    pii_summary = registry.summary()

    # ── Step A ───────────────────────────────────────────────────────────────
    extraction = extract_claims(text_client, document_id, domain, masked_text)

    if not extraction.claims:
        return ComplianceReport(
            document_id=document_id,
            domain=domain,
            claim_verdicts=[],
            overall_risk_note="No checkable claims could be extracted from the input.",
        )

    # ── Steps B + C + D ──────────────────────────────────────────────────────
    claim_verdicts = []

    for claim in extraction.claims:
        try:
            groundings = [
                ground_claim_against_image(vision_client, claim, img, img_id)
                for img_id, img in masked_images.items()
            ] if claim.requires_visual_evidence else []

            verdict = synthesize_verdict(claim, groundings)
            verdict = generate_explanation(text_client, verdict, groundings)

            # Restore real values in explanation and claim text
            verdict.explanation = registry.restore(verdict.explanation or "")
            verdict.claim_text = registry.restore(verdict.claim_text)

            claim_verdicts.append(verdict)

        except Exception as e:
            print(f"Claim {claim.claim_id} failed: {e}")
            claim_verdicts.append(ClaimVerdict(
                claim_id=claim.claim_id,
                claim_text=registry.restore(claim.claim_text),
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                supporting_grounding_ids=[],
                explanation=f"Processing error: {str(e)}"
            ))

    # Build overall risk note including PII summary
    risk_note = None
    if pii_summary["entities_masked"] > 0:
        cats = ", ".join(
            f"{v} {k}(s)" for k, v in pii_summary["categories"].items()
        )
        risk_note = (
            f"PII protection active: {pii_summary['entities_masked']} "
            f"sensitive entities masked before API processing ({cats}). "
            f"All personal data remained local."
        )
    if image_blur_notes:
        total_blurred = sum(len(v) for v in image_blur_notes.values())
        risk_note = (risk_note or "") + (
            f" {total_blurred} sensitive region(s) blurred in images before processing."
        )

    return ComplianceReport(
        document_id=document_id,
        domain=domain,
        claim_verdicts=[v.model_dump() for v in claim_verdicts],
        overall_risk_note=risk_note,
    )