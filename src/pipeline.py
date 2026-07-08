"""
End-to-end orchestrator: document text + images -> ComplianceReport.
Sequential processing — reliable in Streamlit, no threading issues.
Speed comes from Groq's fast inference, not parallelism.
"""

from PIL import Image

from src.extraction.claim_extraction import extract_claims
from src.grounding.image_grounding import ground_claim_against_image
from src.verdict.synthesize import synthesize_verdict
from src.explanation.generate import generate_explanation
from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import ComplianceReport, Domain, ClaimVerdict, Verdict


def run_pipeline(
    text_client: TextModelClient,
    vision_client: VisionModelClient,
    document_id: str,
    domain: Domain,
    document_text: str,
    images: dict[str, Image.Image],
) -> ComplianceReport:

    # Step A — extract atomic claims from document text
    extraction = extract_claims(text_client, document_id, domain, document_text)

    if not extraction.claims:
        return ComplianceReport(
            document_id=document_id,
            domain=domain,
            claim_verdicts=[],
            overall_risk_note="No checkable claims could be extracted from the input."
        )

    claim_verdicts = []

    for claim in extraction.claims:
        try:
            # Step B — ground claim against each uploaded image
            groundings = [
                ground_claim_against_image(vision_client, claim, img, image_id)
                for image_id, img in images.items()
            ] if claim.requires_visual_evidence else []

            # Step C — deterministic verdict from grounding results
            verdict = synthesize_verdict(claim, groundings)

            # Step D — natural language explanation
            verdict = generate_explanation(text_client, verdict, groundings)

            claim_verdicts.append(verdict)

        except Exception as e:
            print(f"Claim {claim.claim_id} failed: {e}")
            claim_verdicts.append(ClaimVerdict(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                supporting_grounding_ids=[],
                explanation=f"Processing error for this claim: {str(e)}"
            ))

    return ComplianceReport(
        document_id=document_id,
        domain=domain,
        claim_verdicts=[v.model_dump() for v in claim_verdicts],
    )