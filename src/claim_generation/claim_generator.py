"""
Claim Generator — Phase 8.
Vision model analyzes evidence, identifies multiple vehicles if present,
pre-fills all inferable fields, generates formal claim document.
"""

import copy
from pydantic import BaseModel, field_validator
from typing import Optional, Union
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.claim_generation.form_schema import CAR_INSURANCE_CLAIM_SCHEMA, FormField


class VehicleInImage(BaseModel):
    vehicle_label: str     # e.g. "Blue sedan on the left", "Dark grey hatchback"
    color: str
    position_in_frame: str # e.g. "left side", "right side", "center"
    damage_visible: bool
    damage_summary: str    # brief one-line summary of damage


class AIPrefilledFields(BaseModel):
    # Multi-vehicle fields
    multiple_vehicles_detected: bool = False
    vehicles_in_image: list[VehicleInImage] = []
    # Standard claim fields — filled ONLY for the selected vehicle
    incident_type: Optional[str] = None
    incident_description: Optional[Union[str, list]] = None
    vehicle_make_model: Optional[str] = None
    vehicle_year: Optional[str] = None
    damage_description: Optional[Union[str, list]] = None
    damage_other_vehicles: Optional[Union[str, list]] = None
    injury_description: Optional[Union[str, list]] = None
    additional_information: Optional[Union[str, list]] = None
    ai_observations: Union[str, list] = ""

    @field_validator(
        "damage_description", "damage_other_vehicles",
        "injury_description", "additional_information",
        "incident_description", "ai_observations",
        mode="before"
    )
    @classmethod
    def coerce_list_to_string(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v


VISION_ANALYSIS_PROMPT = """You are a forensic vehicle damage assessor analyzing 
insurance evidence photographs.

STEP 1 — COUNT AND IDENTIFY ALL VEHICLES IN THE IMAGE:
List every vehicle visible. For each one state:
- vehicle_label: describe by CAMERA FRAME POSITION only, never by vehicle left/right.
  Use: "left side of frame", "right side of frame", "center of frame", "background".
  Include color and make/model if visible.
  Example: "Dark blue BMW (right side of frame)"
  Example: "Red pickup truck (left side of frame, partial view)"
  NEVER say "driver side" or "passenger side" — you cannot determine these from a photo.
- color: vehicle color
- position_in_frame: left side of frame / right side of frame / center / background
- damage_visible: true only if clear physical damage is visible on THIS vehicle
- damage_summary: one specific sentence describing visible damage, or "No damage visible"
  Only describe damage on THIS specific vehicle, not reflections or shadows.

IMPORTANT: Background vehicles that are parked and undamaged should be listed with
damage_visible: false. Only set damage_visible: true if you can clearly see 
deformation, breakage, or displacement on that vehicle.

Set multiple_vehicles_detected to true if more than one vehicle is present.

STEP 2 — GENERAL OBSERVATIONS:
ai_observations: State the camera angle, list ALL vehicles by their frame position,
describe which ones show damage. Be specific about frame position.
Example: "Front angle showing a severely damaged dark blue BMW on the right side of 
the frame. A red truck is partially visible on the left side of the frame with no 
visible damage — appears to be parked nearby."

Leave all other fields as null.

Return JSON matching the required schema exactly.
"""

VEHICLE_DAMAGE_PROMPT = """You are a forensic vehicle damage assessor.
The user has identified their vehicle as: {selected_vehicle}

Based on the image, fill in the following fields FOR THAT SPECIFIC VEHICLE ONLY.
Do not include damage from other vehicles in damage_description.

damage_description: Detailed bullet points for every damaged component on 
{selected_vehicle}. Format each bullet as:
"- [Component name]: [damage type] — [severity] — [functional impact]"
Example:
"- Front left door panel: severely crumpled and buckled inward — severe — 
  structural integrity compromised, door likely non-functional
- Driver side mirror: displaced and misaligned — moderate — 
  visibility impaired
- Side skirt beneath front door: crushed inward approximately 3-4 inches — 
  severe — aerodynamic and structural damage"

Include EVERY damaged component visible on the selected vehicle.
Never include damage from the other vehicle(s).

incident_type: Choose from:
  Head-On Collision / Side Impact Collision / Rear-End Collision /
  Multi-Point Collision / T-Bone Collision / Parking Damage / Unknown

incident_description: Write 3-4 professional sentences starting with:
  "Based on the visible damage pattern, the {selected_vehicle} sustained..."
  Include: which side was impacted, likely direction of force, severity assessment,
  whether the vehicle appears roadworthy, and any safety concerns observed.
  Note: Use camera frame position to describe the vehicle
  (e.g. "the dark blue BMW on the right side of the frame") --
  never say "driver side" or "passenger side" since these cannot
  be confirmed from a photograph alone.

vehicle_make_model: Identify make/model of {selected_vehicle} if possible.
vehicle_year: Estimate year range if identifiable.
damage_other_vehicles: ONLY fill this if another vehicle was DIRECTLY involved 
in the collision with {selected_vehicle}. You need CLEAR visual evidence of 
direct contact between the two vehicles — shared deformation points, overlapping 
damage, or vehicles physically interlocked. If another vehicle is simply parked 
nearby, in the background, or in a parking lot without clear collision evidence, 
write exactly: "No other vehicles confirmed as involved in this incident." 
Never assume a parked or stationary background vehicle is involved in the accident.
injury_description: Note airbag deployment or injury indicators on {selected_vehicle}.
additional_information: Fluid leaks, total loss indicators, towing requirement.
ai_observations: Complete scene description including both vehicles.

Return JSON matching the required schema exactly.
"""


class _ClaimDocument(BaseModel):
    claim_document: str


CLAIM_DOCUMENT_PROMPT = """You are a professional insurance claims writer.
Generate a complete, formal car insurance claim document from the form data below.

In the Damage Assessment section, format EVERY damage point as a numbered claim:

DAMAGE CLAIM POINTS:
1. [Component]: [Description of damage, severity, and functional impact]
2. [Component]: [Description...]

Use ALL damage points from the form data. Do not summarize or skip any.
Write professionally throughout.
For empty fields write: "Not provided by claimant."
End with a Declaration statement.

Form data:
{form_data}

Return JSON with a single key "claim_document" containing the complete formatted text.
"""


def analyze_scene(
    vision_client: VisionModelClient,
    image: Image.Image,
) -> AIPrefilledFields:
    """Step 1: Identify all vehicles in the scene."""
    result = vision_client.structured_call(
        prompt_parts=[VISION_ANALYSIS_PROMPT, image],
        response_schema=AIPrefilledFields,
    )
    return result


def prefill_for_selected_vehicle(
    vision_client: VisionModelClient,
    image: Image.Image,
    selected_vehicle: str,
    base_result: AIPrefilledFields,
) -> tuple[list[FormField], str]:
    """Step 2: Fill claim fields for the user-selected vehicle."""

    class _DamageFields(BaseModel):
        incident_type: Optional[str] = None
        incident_description: Optional[Union[str, list]] = None
        vehicle_make_model: Optional[str] = None
        vehicle_year: Optional[str] = None
        damage_description: Optional[Union[str, list]] = None
        damage_other_vehicles: Optional[Union[str, list]] = None
        injury_description: Optional[Union[str, list]] = None
        additional_information: Optional[Union[str, list]] = None
        ai_observations: Union[str, list] = ""

        @field_validator(
            "damage_description", "damage_other_vehicles",
            "injury_description", "additional_information",
            "incident_description", "ai_observations",
            mode="before"
        )
        @classmethod
        def coerce(cls, v):
            if isinstance(v, list):
                return "\n".join(str(i) for i in v)
            return v

    prompt = VEHICLE_DAMAGE_PROMPT.format(selected_vehicle=selected_vehicle)
    result = vision_client.structured_call(
        prompt_parts=[prompt, image],
        response_schema=_DamageFields,
    )

    schema = copy.deepcopy(CAR_INSURANCE_CLAIM_SCHEMA)
    ai_values = {
        "incident_type":          result.incident_type,
        "incident_description":   result.incident_description,
        "vehicle_make_model":     result.vehicle_make_model,
        "vehicle_year":           result.vehicle_year,
        "damage_description":     result.damage_description,
        "damage_other_vehicles":  result.damage_other_vehicles,
        "injury_description":     result.injury_description,
        "additional_information": result.additional_information,
    }

    for field in schema:
        if field.key in ai_values and ai_values[field.key]:
            field.value     = ai_values[field.key]
            field.ai_filled = True

    return schema, result.ai_observations


def generate_claim_document(
    text_client: TextModelClient,
    schema: list[FormField],
) -> str:
    form_data_lines = []
    current_section = ""
    for field in schema:
        if field.section != current_section:
            current_section = field.section
            form_data_lines.append(f"\n[{current_section}]")
        value = field.value or "UNKNOWN"
        form_data_lines.append(f"{field.label}: {value}")

    prompt = CLAIM_DOCUMENT_PROMPT.format(
        form_data="\n".join(form_data_lines)
    )
    result = text_client.structured_call(
        prompt=prompt,
        response_schema=_ClaimDocument,
    )
    return result.claim_document


def build_validation_text(schema: list[FormField]) -> str:
    """
    Build focused validation text from schema damage fields only.
    Excludes additional_information and negative observations
    (No fluid leaks, No airbag deployment etc) since these are
    observations not verifiable claims.
    """
    lines = []

    incident_type = next(
        (f.value for f in schema if f.key == "incident_type" and f.value), ""
    )
    incident_desc = next(
        (f.value for f in schema if f.key == "incident_description" and f.value), ""
    )
    damage_desc = next(
        (f.value for f in schema if f.key == "damage_description" and f.value), ""
    )
    vehicle = next(
        (f.value for f in schema if f.key == "vehicle_make_model" and f.value), ""
    )
    injuries = next(
        (f.value for f in schema if f.key == "injury_description" and f.value), ""
    )

    # Negative prefixes that indicate observations not claims
    negative_prefixes = (
        "no ", "none", "not visible", "not observed",
        "no fluid", "no airbag", "no other", "no damage",
        "no visible", "no sign", "no indication",
    )

    def is_positive_claim(line: str) -> bool:
        stripped = line.strip().lower()
        if not stripped:
            return False
        return not any(stripped.startswith(p) for p in negative_prefixes)

    if vehicle:
        lines.append(f"Vehicle: {vehicle}")
    if incident_type:
        lines.append(f"Type of Claim: {incident_type}")
    if incident_desc:
        lines.append(f"Incident Description: {incident_desc}")

    if damage_desc:
        # Filter out negative observations — only keep positive damage claims
        positive_damage_lines = [
            line for line in damage_desc.split("\n")
            if is_positive_claim(line)
        ]
        if positive_damage_lines:
            lines.append(f"\nDamage Claims:\n" + "\n".join(positive_damage_lines))

    # Only include injury claims if they are positive (injury actually occurred)
    if injuries and is_positive_claim(injuries):
        lines.append(f"\nInjury Claims: {injuries}")

    # additional_information is intentionally excluded entirely
    # It contains observations (fluid leaks, towing) not verifiable visual claims

    return "\n".join(lines)

class CaseImprovementSuggestions(BaseModel):
    overall_assessment: str
    suggestions: list[str]
    evidence_gaps: list[str]
    recommended_next_steps: list[str]


SUGGESTION_PROMPT = """You are an experienced insurance claims advisor reviewing 
a completed claim form and its validation results.

Based on the claim data and validation verdicts below, provide specific, actionable 
suggestions to help the claimant strengthen their case.

RULES:
- Only suggest things that are genuinely missing or weak based on the actual data
- Never invent problems that aren't evidenced by the verdicts
- Be specific — say exactly which photo angles or documents would help
- If a claim is already Supported with high confidence, do not suggest changes for it
- Focus on claims that are Insufficient Evidence or Missing Expected Evidence
- Keep suggestions practical and achievable for a regular person

Claim Summary:
{claim_summary}

Validation Results:
{validation_results}

Return JSON with:
- overall_assessment: one paragraph honestly assessing the strength of this claim
- suggestions: list of specific things the claimant should add or photograph
- evidence_gaps: list of specific regions/components not visible in submitted evidence
- recommended_next_steps: ordered list of what to do next to strengthen the claim
"""


def generate_case_suggestions(
    text_client: TextModelClient,
    schema: list[FormField],
    claim_verdicts: list,
) -> CaseImprovementSuggestions:
    """
    Analyzes completed claim + validation verdicts and suggests improvements.
    Only called after validation is complete.
    """
    # Build claim summary
    damage = next(
        (f.value for f in schema if f.key == "damage_description" and f.value), ""
    )
    vehicle = next(
        (f.value for f in schema if f.key == "vehicle_make_model" and f.value), ""
    )
    incident = next(
        (f.value for f in schema if f.key == "incident_description" and f.value), ""
    )

    claim_summary = f"Vehicle: {vehicle}\nIncident: {incident}\nDamage claimed:\n{damage}"

    # Build validation results summary
    verdict_lines = []
    for cv in claim_verdicts:
        if isinstance(cv, dict):
            verdict    = cv.get("verdict", "")
            claim_text = cv.get("claim_text", "")
            confidence = cv.get("confidence", 0)
            explanation = cv.get("explanation", "")
        else:
            verdict     = cv.verdict.value
            claim_text  = cv.claim_text
            confidence  = cv.confidence
            explanation = cv.explanation or ""

        verdict_lines.append(
            f"- [{verdict}] ({confidence:.0%}): {claim_text}\n  Reason: {explanation}"
        )

    validation_results = "\n".join(verdict_lines)

    prompt = SUGGESTION_PROMPT.format(
        claim_summary=claim_summary,
        validation_results=validation_results,
    )

    try:
        result = text_client.structured_call(
            prompt=prompt,
            response_schema=CaseImprovementSuggestions,
        )
        return result
    except Exception as e:
        return CaseImprovementSuggestions(
            overall_assessment="Could not generate suggestions at this time.",
            suggestions=[],
            evidence_gaps=[],
            recommended_next_steps=["Consult your insurance company directly."],
        )