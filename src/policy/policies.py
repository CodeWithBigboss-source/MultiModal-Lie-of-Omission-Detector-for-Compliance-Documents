"""
Policy text store.
Only coverage-relevant sections are stored — exclusions, coverage conditions,
claims requirements. Privacy policy, complaints procedure etc are excluded
since they are never relevant to claim validation.
"""

POLICIES = {
    "ireland_aig": {
        "label": "Ireland — AIG Car Insurance",
        "insurer": "AIG Europe S.A.",
        "currency": "EUR",
        "excess_base": 300,
        "text": """
## AIG IRELAND CAR INSURANCE — COVERAGE SUMMARY

### Section 1A — Loss or Damage to Your Car (Collision/Accidental Damage)

COVERED:
- Loss or damage to your car including accessories and spare parts
- Cost of repairing damage to your car
- Cash equivalent to value of loss or damage
- Cost of replacing your car with one of similar type and condition
- Maximum payment is market value immediately prior to loss or damage
- Reasonable cost of taking your car to nearest suitable repairer
- Towage and storage fees up to EUR 300 provided you notify within 48 hours

NEW CAR REPLACEMENT:
- If within 12 months of first registration AND you are first and only registered keeper
- Your car is stolen and not recovered OR
- Cost of repair exceeds 60% of list price inclusive of taxes
- Recorded mileage must not have exceeded 12,000 miles or 18,000 kilometres

NOT COVERED under Section 1A:
- The first EUR 300 of any claim (compulsory excess)
- Additional excess of EUR 300 if driver aged 21 to 24
- Additional excess of EUR 200 if driver aged 25+ with full licence for less than 12 months
- Additional excess of EUR 300 if driver holds Irish Provisional Licence/Learner Permit
- Damage to tyres caused by braking, punctures, cuts or bursts
- Loss or damage where windows are left open or doors left unlocked
- Mechanical, electrical, electronic or computer fault, failure, malfunction or breakdown
- Loss of use or consequential loss of any kind
- Any reduction in market value of your car following repair
- Wear and tear or depreciation
- Part of cost of repair which improves your car beyond its condition before loss
- Cost of parts in excess of manufacturer's last list price
- Loss or damage as a result of fraud or trickery
- Loss or damage caused by moth, vermin, insects or domestic pets
- Loss or damage arising from filling with wrong fuel
- Loss or damage from use of substandard or contaminated fuel, lubricants or parts
- Loss or damage arising from accident where driver convicted or prosecution pending under Road Traffic legislation relating to alcohol or drugs
- Any gradually operating cause

### Section 1B — Fire, Lightning, Theft or Attempted Theft

COVERED:
- Loss or damage to your car if damaged by fire, lightning, theft or attempted theft

NOT COVERED under Section 1B:
- The first EUR 300 of any claim
- Loss or damage from theft or attempted theft if keys have been left unsecured or in/on/near your car whilst unattended
- Loss or damage where windows are left open or doors left unlocked
- Loss of use or consequential loss of any kind
- Wear and tear or depreciation

### Section 3A — Glass in Windscreens and Windows

COVERED:
- Cost of repair or replacement of windscreen or windows
- Repair of any resulting scratching to surrounding bodywork
- If only damage claimed, no claims discount not affected

NOT COVERED:
- Loss or damage to sunroofs
- Any amount over EUR 225 if repair/replacement not carried out by approved glass replacement company

### Section 3B — Personal Accident

COVERED:
- Lump sum of EUR 10,000 per person if you or your spouse accidentally injured in accident
- Covers death, total/permanent loss of sight in one or both eyes, total loss of limb(s)

NOT COVERED:
- Injured person over age 75
- Intentional injury, suicide or attempted suicide
- Driver convicted or prosecution pending for alcohol or drugs under Road Traffic legislation

### Section 3C — Personal Belongings

COVERED:
- Accidental loss or damage to personal belongings while in or on your car
- Maximum EUR 150 per claim

NOT COVERED:
- Mobile telephones
- Money, stamps, tickets, documents and securities
- Goods or equipment carried in connection with trade or business
- Theft if car left unattended and unlocked, keys left in car, or window left open

### Section 3D — Medical Expenses

COVERED:
- Medical expenses from injuries suffered in an accident while in your car
- Maximum EUR 150 per person injured

### Section 3E — Fire Brigade Charges

COVERED:
- Charges levied by fire authority up to EUR 1,500 per accident

### Replacement Locks

COVERED:
- Replacement of door locks, boot lock, ignition/steering lock, lock transmitter
- Maximum EUR 500
- Only if car keys or lock transmitter stolen and identity of garaging address known to persons in receipt of keys

### General Conditions — Claims Requirements

YOU MUST:
- Notify insurer as soon as reasonably possible with full details of incident
- Notify Gardai as soon as aware of any insured property lost or stolen
- Report vandalism or theft to police and obtain crime report number
- Forward every claim form, writ, summons, legal document to insurer unanswered
- Provide all necessary information and assistance required
- NOT admit liability or make offer of payment without written consent
- Provide all reasonable evidence to support your claim
- Maintain your car in safe and roadworthy condition with valid NCT certificate
- Notify insurer within 48 hours for towage and storage fees

### General Exceptions — Not Covered Under Any Section

- Car driven for purposes not specified in Certificate of Motor Insurance
- Driver does not hold a valid licence or is disqualified
- Car driven without your permission
- Car driven in unsafe condition
- Deliberate, wilful or malicious acts by you or an insured person
- Faulty workmanship, defective design or use of defective materials
- War, invasion, civil war, rebellion, military force
- Terrorism
- Pollution and contamination
- Radioactive contamination
- Earthquake
- Riot or civil commotion outside Republic of Ireland
- Airside on any airport or airfield
- Car driven over the alcohol or drug limit
"""
    },

    "uk_axa": {
        "label": "UK — AXA Car Insurance",
        "insurer": "AXA Insurance UK plc",
        "currency": "GBP",
        "excess_base": None,  # variable, shown in schedule
        "text": """
## AXA UK CAR INSURANCE — COVERAGE SUMMARY

### Part A Section 1 — Loss or Damage to Your Car

COVERED:
- Repair of damage to your car, accessories or spare parts
- Replacement of what is lost or damaged and too expensive to repair
- Cash payment for cost of loss or damage
- Reasonable cost of protecting your car and taking it to nearest recommended repairer
- After repair, reasonable cost of delivering car to your UK address
- Market value if car not recovered after theft or beyond economical repair
- Accessories and spare parts in your private garage at time of loss

COURTESY CAR:
- Provided while your car undergoes repair following a claim
- NOT available if car is beyond economical repair
- NOT available if car stolen and not recovered
- NOT available if recommended repairer not used
- NOT available for losses outside the UK

NEW CAR REPLACEMENT:
- Within one year of first registration as new
- Stolen and not recovered OR
- Repairs cost more than 60% of manufacturer's price list including taxes and accessories
- You must own the car
- A replacement must be available

NOT COVERED under Part A Section 1:
- Mechanical or electrical breakdown or failure
- Computer failure (but accidents caused as a result are covered)
- Damage caused by wear and tear, rust, corrosion or gradual deterioration
- Damage caused by an excluded driver
- Damage when car used for excluded purposes
- Loss or damage caused by deception or fraud
- Loss or damage caused by a cyber incident or cyber act to computer systems
- Any excess that applies to the policy
- Loss of use, loss of earnings, or any other consequential loss
- Damage to tyres from braking, punctures, cuts or bursts unless caused by an insured incident

### Part A Section 2 — Glass Damage

COVERED:
- Repair or replacement of glass in windows or windscreens including panoramic windscreens
- Scratching of bodywork caused by glass breaking
- If only damage claimed, no claims discount not affected

NOT COVERED:
- Excess shown in policy schedule
- Sunroofs and panoramic sunroofs where roof glass is separate unit to windscreen glass
- More than GBP 175 if approved repairer not used

### Part A Section 3 — Audio Visual Equipment

COVERED:
- Permanently fitted in-car navigational equipment, car phones, radios, CD players, cassette players, games consoles
- Unlimited cover if fitted by manufacturer as standard specification
- Maximum GBP 500 if not standard manufacturer fit

NOT COVERED:
- Removable or portable equipment able to be used whilst not attached to car

### Part A Section 4 — Replacement Locks

COVERED:
- Door and boot locks, ignition and steering locks, lock transmitter, entry card
- Maximum GBP 1,000
- Only if keys, lock transmitter or entry card lost or stolen
- MUST report loss to police within 24 hours of discovering loss

NOT COVERED:
- Theft excess shown in schedule
- Any amount over GBP 1,000

### Part A Section 5 — Medical Expenses

COVERED:
- Medical expenses for anyone injured in your car during an insured incident

### Part A — What is NOT Covered (Exclusions)

NOT COVERED:
- Mechanical or electrical breakdown or computer failure
- Damage caused by wear and tear, rust, corrosion or gradual deterioration
- Damage to tyres from braking, punctures, cuts or bursts (unless caused by insured incident)
- Any excess shown in policy schedule
- Loss of use or any other consequential loss
- Losses due to deception or fraud
- Cyber incidents affecting computer systems
- Damage from misfuelling outside UK
- Only 10 litres of correct fuel covered for misfuelling

### Claims Conditions — What You Must Do

YOU MUST IMMEDIATELY:
- Call claims helpline 0345 608 0230
- Do whatever you can to protect the car and its accessories
- Take all reasonable steps to recover missing property
- Provide full details of any other party involved
- Send any letters and documents received before replying to them
- If you know of any future prosecution or coroner's inquest, tell insurer immediately in writing
- If requested, send written details of your claim within 31 days
- NOT admit anything or make any offer or promise about a claim without written permission

### General Exclusions — Not Covered Under Any Section

- Car used for purposes not specified in certificate of insurance
- Driver not covered by certificate of insurance
- Driver you know has no driving licence, is disqualified, or is prevented by law from holding licence
- Driver failing to meet conditions of their licence
- Car towing a caravan, trailer or other vehicle for payment
- Car being used on a track or roadway designed for track use or vehicle performance activities
- Airside on any airport or airfield premises
- Liability arising from agreement or contract unless liability exists anyway
- War, invasion, civil war, rebellion, military force or coup
- Radioactive contamination or nuclear assemblies
- Earthquake
- Riot or civil commotion outside England, Scotland, Wales, Isle of Man or Channel Islands
- Driver over alcohol limit or unfit through drink or drugs
- Driver failing to provide breath, blood or urine sample without lawful reason
- International sanctions
"""
    }
}


def get_policy_labels() -> dict[str, str]:
    return {key: val["label"] for key, val in POLICIES.items()}


def get_policy_text(policy_key: str) -> str:
    return POLICIES[policy_key]["text"]


def get_policy_info(policy_key: str) -> dict:
    p = POLICIES[policy_key]
    return {
        "label":    p["label"],
        "insurer":  p["insurer"],
        "currency": p["currency"],
    }

def generate_policy_pdf(policy_key: str) -> bytes:
    """Generate a downloadable PDF of the selected policy."""
    from fpdf import FPDF

    policy = POLICIES[policy_key]

    class PolicyPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 12)
            self.set_fill_color(10, 40, 80)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, f"  {policy['label']} — Policy Document", fill=True, ln=True)
            self.set_text_color(0, 0, 0)
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()} | {policy['insurer']}", align="C")

    def sanitize(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        replacements = {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
        }
        for char, rep in replacements.items():
            text = text.replace(char, rep)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = PolicyPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    for line in policy["text"].split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_fill_color(10, 40, 80)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 9, sanitize(f"  {line[3:]}"), fill=True, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(40, 80, 140)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, sanitize(f"  {line[4:]}"), fill=True, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        elif line.startswith("COVERED:") or line.startswith("NOT COVERED"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(230, 240, 255)
            pdf.cell(0, 7, sanitize(line), fill=True, ln=True)
            pdf.set_font("Helvetica", "", 9)
        elif line.startswith("- ") or line.startswith("YOU MUST"):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(18)
            pdf.multi_cell(177, 5, sanitize(line))
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(15)
            pdf.multi_cell(180, 5, sanitize(line))

    return bytes(pdf.output())