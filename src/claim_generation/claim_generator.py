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
- vehicle_label: descriptive name (e.g. "Blue sedan on the left")
- color: vehicle color
- position_in_frame: where it is in the frame
- damage_visible: true/false
- damage_summary: one sentence describing its damage or "No damage visible"

Set multiple_vehicles_detected to true if more than one vehicle is present.

STEP 2 — GENERAL OBSERVATIONS:
ai_observations: State the camera angle, list ALL vehicles present, describe the 
accident scene. Include which vehicles show damage and how severe. Be specific.
Example: "Front-left angle showing a two-vehicle collision on a road surface. 
The blue sedan (left) has severe front-end damage with crumpled hood and displaced 
front bumper. The dark grey vehicle (right) shows severe door panel deformation 
on the driver side with visible structural buckling."

For the remaining fields — leave them ALL as null for now.
They will be filled once the user identifies which vehicle is theirs.

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

vehicle_make_model: Identify make/model of {selected_vehicle} if possible.
vehicle_year: Estimate year range if identifiable.
damage_other_vehicles: Describe damage to the OTHER vehicle(s) in one sentence.
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
    Build focused validation text from schema fields directly.
    Uses bullet-point damage description rather than verbose claim document
    so the pipeline extracts clean, specific claims to validate.
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
    additional = next(
        (f.value for f in schema if f.key == "additional_information" and f.value), ""
    )

    if vehicle:
        lines.append(f"Vehicle: {vehicle}")
    if incident_type:
        lines.append(f"Type of Claim: {incident_type}")
    if incident_desc:
        lines.append(f"Incident Description: {incident_desc}")
    if damage_desc:
        lines.append(f"\nDamage Claims:\n{damage_desc}")
    if injuries:
        lines.append(f"\nInjury Claims: {injuries}")
    if additional:
        lines.append(f"\nAdditional Claims: {additional}")

    return "\n".join(lines)