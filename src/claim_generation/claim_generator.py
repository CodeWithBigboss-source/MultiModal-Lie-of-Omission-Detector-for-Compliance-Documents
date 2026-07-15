"""
Claim Generator — Phase 8.
Vision model analyzes evidence and pre-fills all inferable fields.
Text model generates formal claim document from completed schema.
"""

import copy
from pydantic import BaseModel
from typing import Optional
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.claim_generation.form_schema import CAR_INSURANCE_CLAIM_SCHEMA, FormField
from pydantic import field_validator
from typing import Union

class AIPrefilledFields(BaseModel):
    incident_type: Optional[str] = None
    incident_description: Optional[str] = None
    vehicle_make_model: Optional[str] = None
    vehicle_year: Optional[str] = None
    damage_description: Optional[Union[str, list]] = None
    damage_other_vehicles: Optional[Union[str, list]] = None
    injury_description: Optional[Union[str, list]] = None
    additional_information: Optional[Union[str, list]] = None
    ai_observations: Union[str, list] = ""

    @field_validator(
        "damage_description",
        "damage_other_vehicles",
        "injury_description",
        "additional_information",
        "ai_observations",
        mode="before"
    )
    @classmethod
    def coerce_list_to_string(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v


VISION_PREFILL_PROMPT = """You are a professional insurance damage assessor analyzing 
photographic evidence to pre-fill an insurance claim form.

Study the image thoroughly and fill in every field you can observe.
Be SPECIFIC, DETAILED, and PROFESSIONAL. This is a formal insurance document.

For damage_description: Write detailed bullet points for EVERY damaged component 
you can see. For each damaged part state:
- Which specific part (e.g. front left door panel, front bumper, hood)
- Nature of damage (crumpled, dented, scratched, shattered, buckled, deformed)
- Severity (minor / moderate / severe)
Example format:
"- Front left door panel: severely crumpled and buckled inward, structural deformation visible
- Side mirror (driver side): displaced and misaligned due to door panel impact
- Side skirt / rocker panel: bent and crushed inward beneath the door"

For incident_type: Determine from damage pattern. Options: Collision, Side Impact,
Rear-End Collision, Head-On Collision, Weather Damage, Vandalism, Parking Damage, 
Rollover, Unknown.

For incident_description: Write a professional paragraph describing what likely 
happened based on the damage pattern. Use phrases like "Based on the visible damage 
pattern, it appears the vehicle sustained..."

For vehicle_make_model: Identify the vehicle make and model if visible.
For vehicle_year: Estimate the year range if identifiable.
For damage_other_vehicles: Describe any other vehicles visible in the image.
For injury_description: Note any visible signs of injury or airbag deployment.
For additional_information: Note anything else relevant — airbag deployment, 
fluid leaks, structural integrity concerns.
For ai_observations: Your complete objective description of everything in the image.

CRITICAL: Never leave damage_description, incident_type, or incident_description 
as null if there is ANY visible damage in the image. These are the most important fields.

Return JSON matching the required schema exactly.
"""


CLAIM_DOCUMENT_PROMPT = """You are a professional insurance claims writer.
Generate a formal, complete car insurance claim document based on the following 
completed form data. Write it as a real insurance claim submission.

Use professional language. Be specific and factual.
Format it clearly with section headers.
For fields marked UNKNOWN or empty, write "Not provided by claimant."

Form data:
{form_data}

Return JSON with a single key "claim_document" containing the full formatted claim text.
"""


class _ClaimDocument(BaseModel):
    claim_document: str


def prefill_from_evidence(
    vision_client: VisionModelClient,
    image: Image.Image,
) -> tuple[list[FormField], str]:
    result = vision_client.structured_call(
        prompt_parts=[VISION_PREFILL_PROMPT, image],
        response_schema=AIPrefilledFields,
    )

    schema = copy.deepcopy(CAR_INSURANCE_CLAIM_SCHEMA)

    ai_values = {
        "incident_type":         result.incident_type,
        "incident_description":  result.incident_description,
        "vehicle_make_model":    result.vehicle_make_model,
        "vehicle_year":          result.vehicle_year,
        "damage_description":    result.damage_description,
        "damage_other_vehicles": result.damage_other_vehicles,
        "injury_description":    result.injury_description,
        "additional_information":result.additional_information,
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

    form_data = "\n".join(form_data_lines)
    prompt = CLAIM_DOCUMENT_PROMPT.format(form_data=form_data)

    result = text_client.structured_call(
        prompt=prompt,
        response_schema=_ClaimDocument,
    )
    return result.claim_document