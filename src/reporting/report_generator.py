"""
Compliance Report Generator.
Produces a formatted PDF from a ComplianceReport object.
All text is sanitized to latin-1 before going to FPDF
since Helvetica only supports latin-1 range characters.
"""

from fpdf import FPDF
from datetime import datetime
from src.utils.schemas import ComplianceReport
from src.policy.validator import PolicyValidationReport


def sanitize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Replace common unicode punctuation with ASCII equivalents
    # before encoding — encode("replace") turns them into ? which looks bad
    replacements = {
        "\u2014": "-",   # em dash —
        "\u2013": "-",   # en dash –
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00b7": "-",   # middle dot
        "\u2022": "-",   # bullet
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


VERDICT_ICONS = {
    "Supported":                 "SUPPORTED",
    "Partially Supported":       "PARTIALLY SUPPORTED",
    "Contradicted":              "CONTRADICTED",
    "Insufficient Evidence":     "INSUFFICIENT EVIDENCE",
    "Missing Expected Evidence": "MISSING EXPECTED EVIDENCE",
}

VERDICT_COLORS = {
    "Supported":                 (200, 240, 200),
    "Partially Supported":       (255, 240, 200),
    "Contradicted":              (255, 200, 200),
    "Insufficient Evidence":     (220, 220, 220),
    "Missing Expected Evidence": (255, 210, 210),
}


class CompliancePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 30, 30)
        self.set_text_color(255, 255, 255)
        self.cell(
            0, 12,
            sanitize("  Compliance Evidence Analysis Report"),
            fill=True, ln=True
        )
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        # Use only ASCII-safe separators — no em-dash, no unicode
        footer_text = sanitize(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
            f"Page {self.page_no()}  |  "
            f"Multimodal Lie-of-Omission Detector - Genesys Research Lab"
        )
        self.cell(0, 10, footer_text, align="C")


def generate_pdf_report(
    report: ComplianceReport,
    elapsed_seconds: float = 0
) -> bytes:
    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # ── Meta section ─────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Report Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    meta_rows = [
        ("Reference ID",       str(report.document_id)),
        ("Domain",             str(report.domain).replace("_", " ").title()),
        ("Claims Evaluated",   str(len(report.claim_verdicts))),
        ("Analysis Time",      f"{elapsed_seconds:.1f} seconds"),
        ("Report Generated",   datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    for label, value in meta_rows:
        pdf.cell(60, 7, sanitize(label + ":"), border=0)
        pdf.cell(0,  7, sanitize(value), ln=True)

    # ── PII note ─────────────────────────────────────────────
    if report.overall_risk_note:
        pdf.ln(3)
        pdf.set_fill_color(230, 245, 255)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(
            0, 7,
            sanitize(f"Privacy: {report.overall_risk_note}"),
            fill=True
        )

    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ── Verdict summary table ─────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Verdict Summary", ln=True)
    pdf.ln(2)

    # Table header
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(8,  8, "#",           fill=True, border=1)
    pdf.cell(90, 8, "Claim",       fill=True, border=1)
    pdf.cell(55, 8, "Verdict",     fill=True, border=1)
    pdf.cell(25, 8, "Confidence",  fill=True, border=1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i, cv in enumerate(report.claim_verdicts, 1):
        if isinstance(cv, dict):
            verdict = cv.get("verdict", "")
            claim   = cv.get("claim_text", "")
            conf    = cv.get("confidence", 0)
        else:
            verdict = cv.verdict.value
            claim   = cv.claim_text
            conf    = cv.confidence

        r, g, b = VERDICT_COLORS.get(verdict, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "", 9)

        claim_short = sanitize(claim[:60] + "..." if len(claim) > 60 else claim)
        pdf.cell(8,  8, str(i),                   fill=True, border=1)
        pdf.cell(90, 8, claim_short,               fill=True, border=1)
        pdf.cell(55, 8, sanitize(verdict),         fill=True, border=1)
        pdf.cell(25, 8, f"{conf:.0%}",             fill=True, border=1)
        pdf.ln()

    pdf.ln(8)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ── Detailed findings ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Detailed Findings", ln=True)
    pdf.ln(2)

    for i, cv in enumerate(report.claim_verdicts, 1):
        if isinstance(cv, dict):
            verdict     = cv.get("verdict", "")
            claim       = cv.get("claim_text", "")
            conf        = cv.get("confidence", 0)
            explanation = cv.get("explanation", "")
        else:
            verdict     = cv.verdict.value
            claim       = cv.claim_text
            conf        = cv.confidence
            explanation = cv.explanation or ""

        # Sanitize everything before touching FPDF
        verdict     = sanitize(verdict)
        claim       = sanitize(claim)
        explanation = sanitize(explanation)

        r, g, b = VERDICT_COLORS.get(
            # unsanitized key lookup since VERDICT_COLORS keys are ASCII
            cv.get("verdict", "") if isinstance(cv, dict) else cv.verdict.value,
            (240, 240, 240)
        )

        # Claim header bar
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "B", 10)
        verdict_label = sanitize(
            VERDICT_ICONS.get(
                cv.get("verdict", "") if isinstance(cv, dict) else cv.verdict.value,
                verdict
            )
        )
        pdf.cell(
            0, 9,
            sanitize(f"  Claim {i}: {verdict_label} ({conf:.0%} confidence)"),
            fill=True, ln=True
        )

        # Claim text
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, "Claim:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, claim)

        # Explanation
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(15)
        pdf.cell(0, 7, "Explanation:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, explanation)

        pdf.ln(4)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

    return bytes(pdf.output())

def generate_combined_pdf_report(
    claim_document: str,
    report: ComplianceReport,
    elapsed_seconds: float = 0,
    schema: list = None,
    policy_report=None,
) -> bytes:
    """
    Generates a single PDF containing:
    1. The user's generated claim document
    2. The compliance validation report
    """
    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    SECTION_COLORS = {
        "Policyholder Information": (20,  80,  140),
        "Incident Details":         (140, 50,  20),
        "Vehicle Information":      (20,  100, 60),
        "Damage Assessment":        (100, 20,  100),
    }

    # ── Part 1: Claim Document ────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(10, 40, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 13, "  PART 1 - INSURANCE CLAIM SUBMISSION", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(
        0, 7,
        "  Generated by Multimodal Lie-of-Omission Detector - Genesys Research Lab",
        fill=True, ln=True
    )
    pdf.ln(5)

    if schema:
        # Rich schema-based rendering — mirrors generate_claim_form_pdf
        current_section = ""
        for field in schema:
            if field.section != current_section:
                current_section = field.section
                pdf.ln(4)
                r, g, b = SECTION_COLORS.get(current_section, (50, 50, 50))
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 9, f"  {sanitize(current_section)}", fill=True, ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)

            value = sanitize(field.value or "Not provided")
            label = sanitize(field.label)

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(240, 245, 255)
            pdf.cell(0, 6, label + ":", fill=True, ln=True)

            if field.ai_filled and field.value:
                pdf.set_fill_color(230, 255, 230)
            else:
                pdf.set_fill_color(255, 255, 255)

            pdf.set_font("Helvetica", "", 9)
            if field.field_type == "textarea" or "\n" in value:
                for line in value.split("\n"):
                    if line.strip():
                        pdf.set_x(18)
                        pdf.multi_cell(177, 5, sanitize(line), fill=True)
            else:
                pdf.set_x(18)
                pdf.cell(177, 6, value, fill=True, ln=True)
            pdf.ln(2)

        # Formal claim statement on new page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(10, 40, 80)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  FORMAL CLAIM STATEMENT", fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 9)
        for line in sanitize(claim_document).split("\n"):
            if not line.strip():
                pdf.ln(3)
            elif "DAMAGE CLAIM POINTS" in line:
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_fill_color(100, 20, 100)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(0, 8, f"  {sanitize(line.strip())}", fill=True, ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "", 9)
            elif line.strip().startswith("[") and line.strip().endswith("]"):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_fill_color(230, 240, 255)
                pdf.cell(0, 7, sanitize(line.strip()), fill=True, ln=True)
                pdf.set_font("Helvetica", "", 9)
            elif line.strip()[:2] in [f"{i}." for i in range(1, 20)]:
                pdf.set_fill_color(245, 245, 255)
                pdf.set_x(15)
                pdf.multi_cell(180, 6, sanitize(line), fill=True)
                pdf.ln(1)
            else:
                pdf.set_x(15)
                pdf.multi_cell(180, 6, sanitize(line))

    else:
        # Fallback: plain text rendering if no schema passed
        pdf.set_font("Helvetica", "", 9)
        for line in sanitize(claim_document).split("\n"):
            if not line.strip():
                pdf.ln(3)
            else:
                pdf.set_x(15)
                pdf.multi_cell(180, 6, sanitize(line))

    # ── Divider ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(100, 20, 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "  PART 2 - COMPLIANCE VALIDATION REPORT", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Reuse existing report sections ────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Validation Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 7, "Claims Evaluated:", border=0)
    pdf.cell(0, 7, str(len(report.claim_verdicts)), ln=True)
    pdf.cell(60, 7, "Analysis Time:", border=0)
    pdf.cell(0, 7, f"{elapsed_seconds:.1f} seconds", ln=True)

    if report.overall_risk_note:
        pdf.ln(3)
        pdf.set_fill_color(230, 245, 255)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 7, sanitize(f"Privacy: {report.overall_risk_note}"), fill=True)

    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # Verdict summary table
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(8,  8, "#",          fill=True, border=1)
    pdf.cell(90, 8, "Claim",      fill=True, border=1)
    pdf.cell(55, 8, "Verdict",    fill=True, border=1)
    pdf.cell(25, 8, "Confidence", fill=True, border=1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i, cv in enumerate(report.claim_verdicts, 1):
        if isinstance(cv, dict):
            verdict = cv.get("verdict", "")
            claim   = cv.get("claim_text", "")
            conf    = cv.get("confidence", 0)
            explanation = cv.get("explanation", "")
        else:
            verdict = cv.verdict.value
            claim   = cv.claim_text
            conf    = cv.confidence
            explanation = cv.explanation or ""

        r, g, b = VERDICT_COLORS.get(verdict, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "", 9)
        claim_short = sanitize(claim[:60] + "..." if len(claim) > 60 else claim)
        pdf.cell(8,  8, str(i),             fill=True, border=1)
        pdf.cell(90, 8, claim_short,         fill=True, border=1)
        pdf.cell(55, 8, sanitize(verdict),   fill=True, border=1)
        pdf.cell(25, 8, f"{conf:.0%}",       fill=True, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Detailed Findings", ln=True)
    pdf.ln(2)

    for i, cv in enumerate(report.claim_verdicts, 1):
        if isinstance(cv, dict):
            verdict     = cv.get("verdict", "")
            claim       = cv.get("claim_text", "")
            conf        = cv.get("confidence", 0)
            explanation = cv.get("explanation", "")
        else:
            verdict     = cv.verdict.value
            claim       = cv.claim_text
            conf        = cv.confidence
            explanation = cv.explanation or ""

        verdict     = sanitize(verdict)
        claim       = sanitize(claim)
        explanation = sanitize(explanation)

        r, g, b = VERDICT_COLORS.get(
            cv.get("verdict", "") if isinstance(cv, dict) else cv.verdict.value,
            (240, 240, 240)
        )
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "B", 10)
        verdict_label = sanitize(VERDICT_ICONS.get(
            cv.get("verdict", "") if isinstance(cv, dict) else cv.verdict.value,
            verdict
        ))
        pdf.cell(
            0, 9,
            sanitize(f"  Claim {i}: {verdict_label} ({conf:.0%} confidence)"),
            fill=True, ln=True
        )
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, "Claim:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, claim)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(15)
        pdf.cell(0, 7, "Explanation:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, explanation)
        pdf.ln(4)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)
    
    # Part 3 — Policy assessment (optional)
    if policy_report is not None:
        add_policy_section_to_pdf(pdf, policy_report)

    return bytes(pdf.output())

def generate_claim_form_pdf(
    schema: list,
    claim_document: str,
) -> bytes:
    """
    Generates a styled PDF of the completed claim form
    including AI-generated damage claim points.
    """
    SECTION_COLORS = {
        "Policyholder Information": (20,  80,  140),
        "Incident Details":         (140, 50,  20),
        "Vehicle Information":      (20,  100, 60),
        "Damage Assessment":        (100, 20,  100),
    }

    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Cover title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(10, 40, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "  CAR INSURANCE CLAIM FORM", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 8,
        "  Generated by Multimodal Lie-of-Omission Detector - Genesys Research Lab",
        fill=True, ln=True
    )
    pdf.ln(6)

    current_section = ""
    for field in schema:
        # Section header with color
        if field.section != current_section:
            current_section = field.section
            pdf.ln(4)
            r, g, b = SECTION_COLORS.get(current_section, (50, 50, 50))
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 9, f"  {sanitize(current_section)}", fill=True, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)

        value = sanitize(field.value or "Not provided")
        label = sanitize(field.label)

        # Label
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 245, 255)
        pdf.cell(0, 6, label + ":", fill=True, ln=True)

        # Value — highlight AI-filled fields
        if field.ai_filled and field.value:
            pdf.set_fill_color(230, 255, 230)  # light green for AI fields
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(18)

        if field.field_type == "textarea" or "\n" in value:
            for line in value.split("\n"):
                if line.strip():
                    pdf.set_x(18)
                    pdf.multi_cell(177, 5, sanitize(line), fill=True)
        else:
            pdf.cell(177, 6, value, fill=True, ln=True)

        pdf.ln(2)

    # Formal claim statement
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(10, 40, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 11, "  FORMAL CLAIM STATEMENT", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 9)
    for line in sanitize(claim_document).split("\n"):
        if not line.strip():
            pdf.ln(3)
        elif line.strip().startswith("DAMAGE CLAIM POINTS"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(100, 20, 100)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, f"  {sanitize(line.strip())}", fill=True, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 9)
        elif line.strip().startswith("[") and line.strip().endswith("]"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(230, 240, 255)
            pdf.cell(0, 7, sanitize(line.strip()), fill=True, ln=True)
            pdf.set_font("Helvetica", "", 9)
        elif line.strip()[:2] in [f"{i}." for i in range(1, 20)]:
            # Numbered claim points — highlighted
            pdf.set_fill_color(245, 245, 255)
            pdf.set_x(15)
            pdf.multi_cell(180, 6, sanitize(line), fill=True)
            pdf.ln(1)
        else:
            pdf.set_x(15)
            pdf.multi_cell(180, 6, sanitize(line))

    return bytes(pdf.output())

def add_policy_section_to_pdf(
    pdf: CompliancePDF,
    policy_report: "PolicyValidationReport",
) -> None:
    """
    Adds Part 3 — Policy Compliance Assessment to an existing PDF.
    Called after Part 2 is written in generate_combined_pdf_report.
    """
    DECISION_COLORS = {
        "COVERED":           (200, 240, 200),
        "EXCLUDED":          (255, 200, 200),
        "CONDITIONAL":       (255, 240, 200),
        "INSUFFICIENT_INFO": (220, 220, 220),
    }

    RECOMMENDATION_COLORS = {
        "PROCEED":          (20,  120, 40),
        "LIKELY_REJECTED":  (180, 20,  20),
        "PARTIAL":          (180, 100, 0),
        "NEEDS_MORE_INFO":  (80,  80,  80),
    }

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    r, g, b = RECOMMENDATION_COLORS.get(
        policy_report.overall_recommendation, (50, 50, 50)
    )
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 13, "  PART 3 - POLICY COMPLIANCE ASSESSMENT", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Policy and recommendation header
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 7, "Policy Applied:", border=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, sanitize(policy_report.policy_name), ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 7, "Overall Recommendation:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    r, g, b = RECOMMENDATION_COLORS.get(
        policy_report.overall_recommendation, (50, 50, 50)
    )
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 7, sanitize(policy_report.overall_recommendation), ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_x(15)
    pdf.multi_cell(180, 6, sanitize(policy_report.overall_reasoning), fill=True)
    pdf.ln(4)

    # Critical flags
    if policy_report.critical_flags:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(255, 220, 220)
        pdf.cell(0, 8, "  CRITICAL FLAGS", fill=True, ln=True)
        pdf.set_font("Helvetica", "", 9)
        for flag in policy_report.critical_flags:
            pdf.set_x(15)
            pdf.multi_cell(180, 6, sanitize(f"! {flag}"))
        pdf.ln(4)

    # Per-claim assessments
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Per-Claim Policy Assessment", ln=True)
    pdf.ln(2)

    for i, assessment in enumerate(policy_report.claim_assessments, 1):
        r, g, b = DECISION_COLORS.get(assessment.policy_decision, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(
            0, 9,
            sanitize(f"  Claim {i}: {assessment.policy_decision} -- {assessment.policy_clause_cited}"),
            fill=True, ln=True
        )

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Claim:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 6, sanitize(assessment.claim_text))

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Policy Reasoning:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 6, sanitize(assessment.policy_reasoning))

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Recommended Action:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 6, sanitize(assessment.recommended_action))

        pdf.ln(4)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)