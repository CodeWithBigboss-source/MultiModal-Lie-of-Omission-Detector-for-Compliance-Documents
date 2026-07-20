"""
Policy Validator — Phase 9.

Runs AFTER evidence grounding verdicts are complete.
Takes the claim verdicts + selected policy text and produces
a policy-grounded recommendation per claim.

Design decisions for speed:
- Single Groq text call covering ALL claims at once (not one per claim)
- Only coverage-relevant policy sections passed (not full document)
- Runs in parallel with PDF generation — does not block report display
"""

from pydantic import BaseModel
from typing import Optional
from src.utils.model_client import TextModelClient


class PolicyClaimAssessment(BaseModel):
    claim_text: str
    policy_decision: str          # COVERED / EXCLUDED / CONDITIONAL / INSUFFICIENT_INFO
    policy_clause_cited: str      # exact section name e.g. "Section 1A Exclusion: Wrong fuel"
    policy_reasoning: str         # one sentence explaining why
    recommended_action: str       # what claimant should do next


class PolicyValidationReport(BaseModel):
    policy_name: str
    overall_recommendation: str   # PROCEED / LIKELY_REJECTED / PARTIAL / NEEDS_MORE_INFO
    overall_reasoning: str        # one paragraph summary
    claim_assessments: list[PolicyClaimAssessment]
    critical_flags: list[str]     # high-priority issues the claimant must address


POLICY_VALIDATION_PROMPT = """You are a senior insurance claims assessor.
You have been given:
1. The results of an evidence grounding analysis (what the AI found in the submitted photos)
2. The relevant sections of the insurance policy

Your job is to assess each claim point against the policy and determine whether it
would be COVERED, EXCLUDED, CONDITIONAL, or requires INSUFFICIENT_INFO to decide.

DECISION DEFINITIONS:
- COVERED: The damage/loss described clearly falls within policy coverage
- EXCLUDED: The damage/loss is explicitly excluded by a named policy clause
- CONDITIONAL: Covered in principle but depends on information not yet provided
  (e.g. police report required, excess applies, approved repairer required)
- INSUFFICIENT_INFO: Cannot determine coverage without more information from claimant

RULES:
- Cite the exact policy section for every decision
- Do not invent exclusions not present in the policy text
- Do not approve claims that are explicitly excluded
- Be specific — say "Section 1A Exclusion: Tyre damage from puncture" not just "excluded"
- For CONDITIONAL decisions, state exactly what condition must be met
- overall_recommendation must reflect the combined picture:
  PROCEED = majority of claims appear covered, no critical exclusions
  LIKELY_REJECTED = one or more claims hit a hard exclusion that invalidates the claim
  PARTIAL = some claims covered, some excluded
  NEEDS_MORE_INFO = cannot determine without claimant providing more information

Policy:
{policy_text}

Evidence Grounding Results (what was actually found in the submitted images):
{evidence_results}

Claim Points to Assess:
{claim_points}

Return JSON matching the required schema exactly.
"""


def validate_against_policy(
    text_client: TextModelClient,
    claim_verdicts: list,
    policy_key: str,
    policy_text: str,
) -> PolicyValidationReport:
    """
    Single Groq text call assessing all claims against selected policy.
    Fast — text only, no vision.
    """
    from src.policy.policies import get_policy_info
    policy_info = get_policy_info(policy_key)

    # Build evidence results summary
    evidence_lines = []
    for i, cv in enumerate(claim_verdicts, 1):
        if isinstance(cv, dict):
            verdict     = cv.get("verdict", "")
            claim_text  = cv.get("claim_text", "")
            confidence  = cv.get("confidence", 0)
            explanation = cv.get("explanation", "")
        else:
            verdict     = cv.verdict.value
            claim_text  = cv.claim_text
            confidence  = cv.confidence
            explanation = cv.explanation or ""

        evidence_lines.append(
            f"{i}. [{verdict}] ({confidence:.0%} confidence)\n"
            f"   Claim: {claim_text}\n"
            f"   Evidence finding: {explanation}"
        )

    # Build claim points list
    claim_lines = []
    for i, cv in enumerate(claim_verdicts, 1):
        claim_text = cv.get("claim_text", "") if isinstance(cv, dict) else cv.claim_text
        claim_lines.append(f"{i}. {claim_text}")

    prompt = POLICY_VALIDATION_PROMPT.format(
        policy_text=policy_text,
        evidence_results="\n".join(evidence_lines),
        claim_points="\n".join(claim_lines),
    )

    try:
        result = text_client.structured_call(
            prompt=prompt,
            response_schema=PolicyValidationReport,
        )
        result.policy_name = policy_info["label"]
        return result
    except Exception as e:
        return PolicyValidationReport(
            policy_name=policy_info["label"],
            overall_recommendation="INSUFFICIENT_INFO",
            overall_reasoning=f"Policy validation could not be completed: {e}",
            claim_assessments=[],
            critical_flags=["Policy validation failed — please retry."],
        )