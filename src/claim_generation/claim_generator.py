"""
Claim Generator — Phase 8.

Step 1: Vision model analyzes uploaded evidence and pre-fills
        all AI-inferable fields from the schema.
Step 2: Wizard presents sections one at a time.
        Pre-filled fields shown as confirmed.
        Unknown fields shown as empty inputs.
Step 3: User confirms/edits each section and clicks Next.
Step 4: Complete filled schema generates a formal claim document.
"""

import json
from pydantic import BaseModel
from typing import Optional
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.claim_generation.form_schema import (
    CAR_INSURANCE_CLAIM_SCHEMA,
    FormField,
    get_ai_inferable_fields,
)


# ── What the vision model fills in ───────────────────────────
class AIPrefilledFields(BaseModel):
    incident_type: Optional[str] = None
    incident_description: Optional[str] = None
    vehicle_make_model: Optional[str] = None
    vehicle_year: Optional[str] = None
    damage_description: Optional[str] = None
    damage_other_vehicles: Optional[str] = None
    injury_description: Optional[str] = None
    additional_information: Optional[str] = None
    ai_observations: str = ""   # full raw observation for transparency


VISION_PREFILL_PROMPT = """You are analyzing an insurance evidence photograph to help 
pre-fill a car insurance claim form. Study the image carefully.

From what you can see in this image, fill in as many of the following fields as possible.
Only fill fields you can directly observe from the image.
For fields you cannot determine, leave them as null.
Never invent information not visible in the image.

Fields to fill:
- incident_type: type of claim (Collision / Weather Damage / Vandalism / Other) — infer from damage pattern
- incident_description: describe what likely happened based on the damage pattern you observe
- vehicle_make_model: vehicle make and model if identifiable from the image
- vehicle_year: approximate year if identifiable
- damage_description: comprehensive description of ALL visible damage — be specific about location, severity, type
- damage_other_vehicles: describe damage to any other vehicles visible in image, null if none
- injury_description: describe any visible signs of injury, null if none visible
- additional_information: anything else relevant observed in the image
- ai_observations: your complete, objective description of everything visible in this image

Return JSON matching the required schema exactly.
"""


CLAIM_DOCUMENT_PROMPT = """You are a professional insurance claims writer. 
Generate a formal, complete car insurance claim document based on the following 
filled form data. Write it exactly as a real insurance claim submission would read.
Use professional language. Be specific and factual.

Form data:
{form_data}

Generate the claim as a structured document with clear sections matching the form.
For any field marked as UNKNOWN, write "Not provided" or "To be confirmed."

Return JSON with a single key "claim_document" containing the full formatted claim text.
"""


class _ClaimDocument(BaseModel):
    claim_document: str


def prefill_from_evidence(
    vision_client: VisionModelClient,
    image: Image.Image,
) -> list[FormField]:
    """
    Run vision model against evidence image.
    Returns updated schema with AI-inferable fields pre-filled.
    """
    result = vision_client.structured_call(
        prompt_parts=[VISION_PREFILL_PROMPT, image],
        response_schema=AIPrefilledFields,
    )

    # Create a working copy of the schema
    import copy
    schema = copy.deepcopy(CAR_INSURANCE_CLAIM_SCHEMA)

    # Map AI output back to schema fields
    ai_values = {
        "incident_type":        result.incident_type,
        "incident_description": result.incident_description,
        "vehicle_make_model":   result.vehicle_make_model,
        "vehicle_year":         result.vehicle_year,
        "damage_description":   result.damage_description,
        "damage_other_vehicles":result.damage_other_vehicles,
        "injury_description":   result.injury_description,
        "additional_information": result.additional_information,
    }

    for field in schema:
        if field.key in ai_values and ai_values[field.key]:
            field.value    = ai_values[field.key]
            field.ai_filled = True

    return schema, result.ai_observations


def generate_claim_document(
    text_client: TextModelClient,
    schema: list[FormField],
) -> str:
    """
    Takes the completed schema and generates a formal claim document.
    """
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