"""
End-to-end orchestrator: document text + images -> ComplianceReport.
TextModelClient (Groq) handles Steps A and D.
VisionModelClient (Gemini) handles Step B.
"""

from PIL import Image

from src.extraction.claim_extraction import extract_claims
from src.grounding.image_grounding import ground_claim_against_image
from src.verdict.synthesize import synthesize_verdict
from src.explanation.generate import generate_explanation
from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import ComplianceReport, Domain


def run_pipeline(
    text_client: TextModelClient,
    vision_client: VisionModelClient,
    document_id: str,
    domain: Domain,
    document_text: str,
    images: dict[str, Image.Image],
) -> ComplianceReport:
    # Step A — Groq
    extraction = extract_claims(text_client, document_id, domain, document_text)

    claim_verdicts = []
    for claim in extraction.claims:
        # Step B — Gemini (vision only)
        groundings = [
            ground_claim_against_image(vision_client, claim, img, image_id)
            for image_id, img in images.items()
        ] if claim.requires_visual_evidence else []

        # Step C — deterministic Python rules
        verdict = synthesize_verdict(claim, groundings)

        # Step D — Groq
        verdict = generate_explanation(text_client, verdict, groundings)

        claim_verdicts.append(verdict)

    return ComplianceReport(
        document_id=document_id,
        domain=domain,
        claim_verdicts=[v.model_dump() for v in claim_verdicts],
    )