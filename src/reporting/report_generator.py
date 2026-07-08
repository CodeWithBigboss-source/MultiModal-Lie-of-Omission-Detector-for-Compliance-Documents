"""
Compliance Report Generator — Phase 4a.
Produces a formatted PDF report from a ComplianceReport object.
"""

from fpdf import FPDF
from datetime import datetime
from src.utils.schemas import ComplianceReport


VERDICT_ICONS = {
    "Supported":                 "SUPPORTED",
    "Partially Supported":       "PARTIALLY SUPPORTED",
    "Contradicted":              "CONTRADICTED",
    "Insufficient Evidence":     "INSUFFICIENT EVIDENCE",
    "Missing Expected Evidence": "MISSING EXPECTED EVIDENCE",
}
def sanitize(text: str) -> str:
    """Strip characters outside latin-1 range that FPDF Helvetica cannot render."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class CompliancePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 30, 30)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "  Compliance Evidence Analysis Report", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
            f"Page {self.page_no()}  |  "
            "Multimodal Lie-of-Omission Detector — Genesys Research Lab",
            align="C"
        )


def generate_pdf_report(report: ComplianceReport, elapsed_seconds: float = 0) -> bytes:
    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # ── Meta section ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Report Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 7, "Reference ID:", border=0)
    pdf.cell(0, 7, str(report.document_id), ln=True)
    pdf.cell(60, 7, "Domain:", border=0)
    pdf.cell(0, 7, str(report.domain).replace("_", " ").title(), ln=True)
    pdf.cell(60, 7, "Claims Evaluated:", border=0)
    pdf.cell(0, 7, str(len(report.claim_verdicts)), ln=True)
    pdf.cell(60, 7, "Analysis Time:", border=0)
    pdf.cell(0, 7, f"{elapsed_seconds:.1f} seconds", ln=True)
    pdf.cell(60, 7, "Report Generated:", border=0)
    pdf.cell(0, 7, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True)

    # ── PII note ──────────────────────────────────────────────────────────────
    if report.overall_risk_note:
        pdf.ln(3)
        pdf.set_fill_color(230, 245, 255)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(
            0, 7,
            f"Privacy: {report.overall_risk_note}",
            fill=True
        )

    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ── Verdict summary table ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Verdict Summary", ln=True)
    pdf.ln(2)

    # Table header
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(8,  8, "#",          fill=True, border=1)
    pdf.cell(90, 8, "Claim",      fill=True, border=1)
    pdf.cell(55, 8, "Verdict",    fill=True, border=1)
    pdf.cell(25, 8, "Confidence", fill=True, border=1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    verdict_colors = {
        "Supported":                 (200, 240, 200),
        "Partially Supported":       (255, 240, 200),
        "Contradicted":              (255, 200, 200),
        "Insufficient Evidence":     (220, 220, 220),
        "Missing Expected Evidence": (255, 210, 210),
    }

    for i, cv in enumerate(report.claim_verdicts, 1):
        if isinstance(cv, dict):
            verdict  = cv.get("verdict", "")
            claim    = cv.get("claim_text", "")
            conf     = cv.get("confidence", 0)
        else:
            verdict  = cv.verdict.value
            claim    = cv.claim_text
            conf     = cv.confidence

        r, g, b = verdict_colors.get(verdict, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "", 9)

        claim_short = sanitize(claim[:60] + "..." if len(claim) > 60 else claim)
        verdict     = sanitize(verdict)
        row_h = 8
        pdf.cell(8,  row_h, str(i),              fill=True, border=1)
        pdf.cell(90, row_h, claim_short,          fill=True, border=1)
        pdf.cell(55, row_h, verdict,              fill=True, border=1)
        pdf.cell(25, row_h, f"{conf:.0%}",        fill=True, border=1)
        pdf.ln()

    pdf.ln(8)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ── Detailed findings ─────────────────────────────────────────────────────
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

        # # Sanitize text — remove characters FPDF cannot render
        # def sanitize(text: str) -> str:
        #     return text.encode("latin-1", errors="replace").decode("latin-1")

        claim       = sanitize(claim)
        explanation = sanitize(explanation)

        r, g, b = verdict_colors.get(verdict, (240, 240, 240))

        # Claim header bar
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "B", 10)
        verdict_label = VERDICT_ICONS.get(verdict, verdict)
        pdf.cell(
            0, 9,
            f"  Claim {i}: {verdict_label} ({conf:.0%} confidence)",
            fill=True, ln=True
        )

        # Claim text — always start on fresh line with full width
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, "Claim:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, claim)

        # Explanation text — always start on fresh line with full width
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, "Explanation:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.multi_cell(180, 7, explanation)

        pdf.ln(4)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

        # Claim text
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(30, 7, "Claim:", border=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 7, claim)

        # Explanation
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(30, 7, "Explanation:", border=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 7, explanation)

        pdf.ln(4)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

    return bytes(pdf.output())