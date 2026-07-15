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


VISION_PREFILL_PROMPT = """You are a forensic vehicle damage assessor. Your job is to 
analyze this photograph and produce a detailed, professional damage report.

STEP 1 — IDENTIFY CAMERA ANGLE:
Which side of the vehicle is shown? (front-left, left side, rear-right, etc.)
List ONLY components physically within the camera frame.
Explicitly state what is NOT visible.

STEP 2 — DOCUMENT ALL DAMAGE:
For every damaged component you can see, write one bullet point stating:
  - Exact component name (e.g. front left door panel, hood, front bumper)
  - Type of damage (crumpled, dented, scratched, buckled, shattered, deployed)
  - Severity (minor / moderate / severe)

Do NOT write vague observations like "the vehicle appears damaged."
Write specific findings like:
  "- Front left door panel: severely crumpled and buckled inward with visible structural deformation"
  "- Driver side mirror: displaced downward and misaligned due to door impact"
  "- Side skirt beneath front door: crushed inward approximately 3-4 inches"

STEP 3 — FILL ALL FIELDS:
damage_description: paste your complete bullet-point damage list from Step 2.
  This MUST be a detailed multi-line string with one bullet per damaged component.
  NEVER leave this null if any damage is visible.

incident_type: infer from damage pattern:
  - Door/side damage → "Side Impact Collision"
  - Front damage → "Head-On Collision" or "Front Impact Collision"
  - Rear damage → "Rear-End Collision"
  - Multiple areas → "Multi-Point Collision"
  - No damage visible → "Unknown"

incident_description: write one professional paragraph starting with:
  "Based on the visible damage pattern, the vehicle appears to have sustained..."
  Describe what likely happened. Be specific about which side was impacted and severity.

vehicle_make_model: identify make and model if badges or body shape are recognizable.
  If uncertain, write the body type (e.g. "Compact hatchback, make unidentified").

vehicle_year: estimate year range from body style (e.g. "2015-2020 approximate").

damage_other_vehicles: describe any other vehicles visible in the image.
  If none, write "No other vehicles visible in submitted evidence."

injury_description: note airbag deployment, blood, or other injury indicators.
  If none visible, write "No visible injury indicators in submitted evidence."

additional_information: note anything else — fluid leaks, structural concerns,
  towing required, total loss indicators.

ai_observations: combine Step 1 AND Step 2 into one complete summary paragraph.
  State: camera angle, what components are in frame, what damage is visible on 
  each component, severity, and what is NOT visible. 
  Example: "The image shows the front-left side of a white compact vehicle.
  The front left door panel is severely crumpled and buckled inward with visible 
  structural deformation. The driver side mirror is displaced. The side skirt 
  beneath the door is crushed inward. The rear of the vehicle and the right side 
  are not visible in this frame."
  This must be a complete, specific damage narrative — NOT just a frame description.

CRITICAL RULES:
- damage_description MUST contain specific bullet points for every damaged part
- Never write "appears to be" or "possibly" for clearly visible damage
- Never confuse the car's left/right with the camera's left/right
- State camera angle first, then describe damage relative to camera angle

Return JSON matching the required schema exactly.
"""

CLAIM_DOCUMENT_PROMPT = """You are a professional insurance claims writer.
Generate a complete, formal car insurance claim document from the form data below.

IMPORTANT: The Damage Assessment section must be written as structured claim points.
Each damage item must be a numbered claim point with full explanation.
Format the damage section like this:

DAMAGE CLAIM POINTS:
1. [Component Name]: [Detailed description of damage, severity, and impact on vehicle function]
2. [Component Name]: [Detailed description...]
...

Write the full document with all sections.
Use professional language throughout.
For fields marked UNKNOWN, write "To be provided by claimant."
Include a declaration statement at the end.

Form data:
{form_data}

Return JSON with a single key "claim_document" containing the complete formatted claim text.
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