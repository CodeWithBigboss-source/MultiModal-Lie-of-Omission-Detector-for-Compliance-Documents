"""
End-to-end orchestrator: document text + images -> ComplianceReport.
"""

from PIL import Image

from src.extraction.claim_extraction import extract_claims
from src.grounding.image_grounding import ground_claim_against_image
from src.verdict.synthesize import synthesize_verdict
from src.explanation.generate import generate_explanation
from src.utils.model_client import ModelClient
from src.utils.schemas import ComplianceReport, Domain


def run_pipeline(
    client: ModelClient,
    document_id: str,
    domain: Domain,
    document_text: str,
    images: dict[str, Image.Image],
) -> ComplianceReport:
    extraction = extract_claims(client, document_id, domain, document_text)

    claim_verdicts = []
    for claim in extraction.claims:
        groundings = [
            ground_claim_against_image(client, claim, img, image_id)
            for image_id, img in images.items()
        ] if claim.requires_visual_evidence else []

        verdict = synthesize_verdict(claim, groundings)
        verdict = generate_explanation(client, verdict, groundings)
        claim_verdicts.append(verdict)

    return ComplianceReport(
        document_id=document_id,
        domain=domain,
        claim_verdicts=claim_verdicts,
    )