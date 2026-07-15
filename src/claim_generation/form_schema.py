"""
Claim Form Schema — extracted from real insurance claim form templates.

Every field has:
- key: internal identifier
- label: exactly as it appears on the real form
- section: which section of the form it belongs to
- field_type: text / date / yes_no / number / textarea
- ai_inferable: whether the vision model can attempt to fill this from evidence
- required: whether the form requires this field
- placeholder: hint shown to user in the wizard
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FormField:
    key: str
    label: str
    section: str
    field_type: str          # text / date / yes_no / number / textarea
    ai_inferable: bool       # True = AI will try to fill from evidence
    required: bool = True
    placeholder: str = ""
    value: Optional[str] = None   # filled by AI or user
    ai_filled: bool = False       # True = AI pre-filled this
    user_confirmed: bool = False  # True = user confirmed or edited


# Complete field schema combining both uploaded claim forms
CAR_INSURANCE_CLAIM_SCHEMA: list[FormField] = [

    # ── Section 1: Policyholder Information ──────────────────
    FormField(
        key="first_name",
        label="First Name",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        placeholder="Enter your first name",
    ),
    FormField(
        key="last_name",
        label="Last Name",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        placeholder="Enter your last name",
    ),
    FormField(
        key="policy_number",
        label="Policy Number",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        placeholder="Enter your insurance policy number",
    ),
    FormField(
        key="address_street",
        label="Street Address",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter your street address",
    ),
    FormField(
        key="address_city",
        label="City",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter your city",
    ),
    FormField(
        key="address_state",
        label="State / Province",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter your state or province",
    ),
    FormField(
        key="address_postal",
        label="Postal / Zip Code",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter postal code",
    ),
    FormField(
        key="phone_number",
        label="Contact Phone Number",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        placeholder="Enter a valid phone number",
    ),
    FormField(
        key="email",
        label="Email",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="example@example.com",
    ),
    FormField(
        key="date_of_birth",
        label="Date of Birth",
        section="Policyholder Information",
        field_type="date",
        ai_inferable=False,
        required=False,
        placeholder="MM/DD/YYYY",
    ),
    FormField(
        key="occupation",
        label="Occupation",
        section="Policyholder Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter your occupation",
    ),

    # ── Section 2: Incident Details ───────────────────────────
    FormField(
        key="incident_date",
        label="Date of Incident",
        section="Incident Details",
        field_type="date",
        ai_inferable=False,
        placeholder="MM/DD/YYYY",
    ),
    FormField(
        key="incident_time",
        label="Time of Incident",
        section="Incident Details",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="HH:MM (approximate)",
    ),
    FormField(
        key="incident_location",
        label="Location of Incident",
        section="Incident Details",
        field_type="textarea",
        ai_inferable=False,
        placeholder="Describe where the incident occurred",
    ),
    FormField(
        key="incident_type",
        label="Type of Claim",
        section="Incident Details",
        field_type="text",
        ai_inferable=True,
        placeholder="e.g. Collision, Theft, Weather Damage",
    ),
    FormField(
        key="incident_description",
        label="Description of Incident",
        section="Incident Details",
        field_type="textarea",
        ai_inferable=True,
        placeholder="Describe what happened leading up to and during the incident",
    ),
    FormField(
        key="police_report_filed",
        label="Police Report Filed",
        section="Incident Details",
        field_type="yes_no",
        ai_inferable=False,
        required=False,
        placeholder="Yes or No",
    ),
    FormField(
        key="police_report_number",
        label="Police Report Number",
        section="Incident Details",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter police report number if available",
    ),

    # ── Section 3: Vehicle Information ────────────────────────
    FormField(
        key="vehicle_make_model",
        label="Make and Model",
        section="Vehicle Information",
        field_type="text",
        ai_inferable=True,
        placeholder="e.g. Toyota Corolla",
    ),
    FormField(
        key="vehicle_year",
        label="Year of Manufacture",
        section="Vehicle Information",
        field_type="text",
        ai_inferable=True,
        placeholder="e.g. 2019",
    ),
    FormField(
        key="vehicle_vin",
        label="Vehicle Identification Number (VIN)",
        section="Vehicle Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter VIN if known",
    ),
    FormField(
        key="license_plate",
        label="License Plate Number",
        section="Vehicle Information",
        field_type="text",
        ai_inferable=False,
        required=False,
        placeholder="Enter license plate number",
    ),
    FormField(
        key="current_mileage",
        label="Current Mileage",
        section="Vehicle Information",
        field_type="number",
        ai_inferable=False,
        required=False,
        placeholder="Enter current mileage",
    ),

    # ── Section 4: Damage Assessment ─────────────────────────
    FormField(
        key="damage_description",
        label="Description of Damage to Your Vehicle",
        section="Damage Assessment",
        field_type="textarea",
        ai_inferable=True,
        placeholder="Describe all visible damage to your vehicle",
    ),
    FormField(
        key="damage_other_vehicles",
        label="Description of Damage to Other Vehicle(s)",
        section="Damage Assessment",
        field_type="textarea",
        ai_inferable=True,
        required=False,
        placeholder="Describe damage to other vehicles if applicable",
    ),
    FormField(
        key="estimated_repair_cost",
        label="Estimated Cost of Repairs ($)",
        section="Damage Assessment",
        field_type="number",
        ai_inferable=False,
        required=False,
        placeholder="Enter estimated repair cost",
    ),
    FormField(
        key="injury_description",
        label="Description of Injuries (if any)",
        section="Damage Assessment",
        field_type="textarea",
        ai_inferable=True,
        required=False,
        placeholder="Describe any injuries sustained",
    ),
    FormField(
        key="medical_facilities",
        label="Medical Facilities Visited",
        section="Damage Assessment",
        field_type="textarea",
        ai_inferable=False,
        required=False,
        placeholder="List any hospitals or clinics visited",
    ),
    FormField(
        key="medical_expenses",
        label="Expenses Incurred for Medical Treatment ($)",
        section="Damage Assessment",
        field_type="number",
        ai_inferable=False,
        required=False,
        placeholder="Enter medical expenses amount",
    ),
    FormField(
        key="additional_information",
        label="Additional Information",
        section="Damage Assessment",
        field_type="textarea",
        ai_inferable=True,
        required=False,
        placeholder="Any other relevant information",
    ),
]

SECTIONS = [
    "Policyholder Information",
    "Incident Details",
    "Vehicle Information",
    "Damage Assessment",
]


def get_fields_for_section(section: str) -> list[FormField]:
    return [f for f in CAR_INSURANCE_CLAIM_SCHEMA if f.section == section]


def get_ai_inferable_fields() -> list[FormField]:
    return [f for f in CAR_INSURANCE_CLAIM_SCHEMA if f.ai_inferable]


def schema_to_dict() -> dict:
    return {f.key: f.value or "" for f in CAR_INSURANCE_CLAIM_SCHEMA}