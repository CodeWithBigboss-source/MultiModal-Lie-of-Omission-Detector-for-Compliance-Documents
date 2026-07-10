"""
End-to-end orchestrator with PII layer and optional cross-document reasoning.

Single document mode: text + images -> ComplianceReport
Multi-document mode:  multiple docs + images -> ComplianceReport
                      + CrossDocumentReport appended to overall_risk_note
"""

import json
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
    additional_documents: dict[str, str] | None = None,
) -> ComplianceReport:
    """
    additional_documents: optional dict of {label: text} for multi-document mode.
    When provided, cross-document contradiction detection runs automatically.
    """

    # ── PII Layer ─────────────────────────────────────────────
    registry = PIIRegistry()
    masked_text = mask_text(document_text, registry)

    # Mask additional documents too
    masked_additional = {}
    if additional_documents:
        for label, text in additional_documents.items():
            masked_additional[label] = mask_text(text, registry)

    # Mask images
    masked_images = {}
    image_blur_notes = {}
    for img_id, img in images.items():
        masked_img, blurred = mask_image(img)
        masked_images[img_id] = masked_img
        if blurred:
            image_blur_notes[img_id] = blurred

    pii_summary = registry.summary()

    # ── Cross-document analysis (Phase 5) ─────────────────────
    cross_doc_summary = None
    if masked_additional and len(masked_additional) >= 1:
        from src.reasoning.cross_document import detect_cross_document_contradictions
        all_docs = {"primary_document": masked_text}
        all_docs.update(masked_additional)
        try:
            cross_report = detect_cross_document_contradictions(
                text_client, all_docs
            )
            if cross_report.contradictions:
                contradiction_lines = []
                for c in cross_report.contradictions:
                    contradiction_lines.append(
                        f"[{c.severity.upper()}] {c.source_a} vs {c.source_b}: "
                        f"{registry.restore(c.explanation)}"
                    )
                cross_doc_summary = (
                    f"CROSS-DOCUMENT ANALYSIS: {len(cross_report.contradictions)} "
                    f"contradiction(s) found across {cross_report.total_documents} "
                    f"documents. " + " | ".join(contradiction_lines)
                )
            else:
                cross_doc_summary = (
                    f"CROSS-DOCUMENT ANALYSIS: No factual contradictions found "
                    f"across {cross_report.total_documents} documents "
                    f"({cross_report.total_claims_extracted} claims checked)."
                )
        except Exception as e:
            cross_doc_summary = f"Cross-document analysis error: {e}"

    # ── Merge all document text for primary claim extraction ───
    full_text = masked_text
    if masked_additional:
        full_text = masked_text + "\n\n" + "\n\n".join(
            f"[{label}]\n{text}"
            for label, text in masked_additional.items()
        )

    # ── Step A ─────────────────────────────────────────────────
    extraction = extract_claims(text_client, document_id, domain, full_text)

    if not extraction.claims:
        return ComplianceReport(
            document_id=document_id,
            domain=domain,
            claim_verdicts=[],
            overall_risk_note=(
                "No checkable claims could be extracted. "
                + (cross_doc_summary or "")
            ),
        )

    # ── Steps B + C + D ────────────────────────────────────────
    claim_verdicts = []

    for claim in extraction.claims:
        try:
            groundings = [
                ground_claim_against_image(vision_client, claim, img, img_id)
                for img_id, img in masked_images.items()
            ] if claim.requires_visual_evidence else []

            verdict = synthesize_verdict(claim, groundings)
            verdict = generate_explanation(text_client, verdict, groundings)

            verdict.explanation = registry.restore(verdict.explanation or "")
            verdict.claim_text  = registry.restore(verdict.claim_text)

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

    # ── Build risk note ────────────────────────────────────────
    notes = []
    if pii_summary["entities_masked"] > 0:
        cats = ", ".join(
            f"{v} {k}(s)" for k, v in pii_summary["categories"].items()
        )
        notes.append(
            f"PII protection: {pii_summary['entities_masked']} entities masked "
            f"({cats}). All personal data remained local."
        )
    if image_blur_notes:
        total = sum(len(v) for v in image_blur_notes.values())
        notes.append(f"{total} sensitive region(s) blurred in images.")
    if cross_doc_summary:
        notes.append(cross_doc_summary)

    return ComplianceReport(
        document_id=document_id,
        domain=domain,
        claim_verdicts=[v.model_dump() for v in claim_verdicts],
        overall_risk_note=" | ".join(notes) if notes else None,
    )