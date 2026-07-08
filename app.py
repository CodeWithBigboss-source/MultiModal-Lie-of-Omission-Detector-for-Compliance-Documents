from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import os
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import Domain
from src.pipeline import run_pipeline
import time
start_time = time.time()
from src.reporting.report_generator import generate_pdf_report
from src.classification.document_classifier import classify_document


st.set_page_config(
    page_title="Compliance Evidence Analyzer",
    page_icon="🔍",
    layout="wide"
)

DOMAIN_LABELS = {
    Domain.VEHICLE_INSURANCE:                "🚗 Vehicle Insurance",
    Domain.HEALTH_INSURANCE:                 "🏥 Health Insurance",
    Domain.LOAN_APPLICATION:                 "💰 Loan Application",
    Domain.EVIDENCE_REVIEW:                  "⚖️ Evidence Review",
    Domain.LICENSING_EMPLOYEE_VERIFICATION:  "🪪 Licensing & Employee Verification",
}

VERDICT_CONFIG = {
    "Supported":                 ("✅", "green",  "Evidence clearly confirms this claim."),
    "Partially Supported":       ("🔶", "orange", "Evidence partially confirms this claim."),
    "Contradicted":              ("❌", "red",    "Evidence directly contradicts this claim."),
    "Insufficient Evidence":     ("⚠️", "gray",   "Evidence is unclear or ambiguous."),
    "Missing Expected Evidence": ("🚫", "red",    "Required region not visible in the image."),
}

# ---------------------------------------------------------------------------
# Session state — tracks domain so we can detect changes and reset
# ---------------------------------------------------------------------------
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = Domain.VEHICLE_INSURANCE
if "report" not in st.session_state:
    st.session_state.report = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    selected = st.selectbox(
        "Claim domain",
        options=list(DOMAIN_LABELS.keys()),
        format_func=lambda d: DOMAIN_LABELS[d],
        index=0,
    )

    # If domain changed → clear previous results and uploaded image
    if selected != st.session_state.selected_domain:
        st.session_state.selected_domain = selected
        st.session_state.report = None
        st.session_state.uploaded_image = None
        st.rerun()

    domain = st.session_state.selected_domain

    st.markdown("---")
    st.markdown("### Verdict guide")
    for verdict, (icon, _, desc) in VERDICT_CONFIG.items():
        st.markdown(f"{icon} **{verdict}** — {desc}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔍 Compliance Evidence Analyzer")
st.markdown(f"**Domain:** {DOMAIN_LABELS[domain]}")
st.markdown("---")

col1, col2 = st.columns([1, 1])

# ---------------------------------------------------------------------------
# Input mode toggle
# ---------------------------------------------------------------------------
input_mode = st.radio(
    "How do you want to provide your claim?",
    options=["Type claim manually", "Upload a document (PDF, DOCX, TXT)"],
    horizontal=True,
    key=f"input_mode_{domain.value}",
)

st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if input_mode == "Type claim manually":
        st.subheader("📄 Your Claim")
        claim_text = st.text_area(
            label="Describe your claim in plain language",
            placeholder="Example: My car's left front door is severely damaged. The rear bumper has minor scratches. The windshield is intact.",
            height=200,
            key=f"claim_{domain.value}",
        )
        doc_images = {}
        extraction_notes = []

    else:
        st.subheader("📂 Upload Document")
        doc_file = st.file_uploader(
            "Upload your claim document",
            type=["pdf", "docx", "txt"],
            key=f"doc_{domain.value}",
        )
        claim_text = ""
        doc_images = {}
        extraction_notes = []

        if doc_file:
            from src.ingestion.document_loader import load_document
            with st.spinner("Extracting document content..."):
                content = load_document(doc_file.read(), doc_file.name)
            claim_text = content.extracted_text
            doc_images = content.extracted_images
            extraction_notes = content.extraction_notes

            st.success(
                f"Extracted from {doc_file.name} "
                f"({content.page_count} page(s), "
                f"{len(doc_images)} embedded image(s))"
            )
            if claim_text:
                # Auto-classify document and suggest domain
                with st.spinner("Classifying document type..."):
                    try:
                        text_client_temp = TextModelClient()
                        classification = classify_document(text_client_temp, claim_text)

                        if classification.confidence > 0.6:
                            st.success(
                                f"📋 Detected: **{classification.document_type.replace('_', ' ').title()}** "
                                f"→ Suggested domain: **{classification.suggested_domain.value.replace('_', ' ').title()}** "
                                f"({classification.confidence:.0%} confidence)"
                            )
                            # Auto-switch domain if different from current
                            if classification.suggested_domain != st.session_state.selected_domain:
                                st.info(
                                    f"💡 Auto-switching domain to "
                                    f"**{classification.suggested_domain.value.replace('_', ' ').title()}** "
                                    f"based on document content. You can override this in the sidebar."
                                )
                                st.session_state.selected_domain = classification.suggested_domain
                                st.rerun()
                        else:
                            st.warning(
                                f"Document type unclear ({classification.document_type}). "
                                f"Please select the correct domain from the sidebar."
                            )
                    except Exception:
                        pass  # Classification is best-effort, don't block the user

                with st.expander("Preview extracted text", expanded=False):
                    st.text(claim_text[:2000] + ("..." if len(claim_text) > 2000 else ""))

            if extraction_notes:
                for note in extraction_notes:
                    st.info(note)

            

    document_id = st.text_input(
        "Reference ID",
        value="claim_001",
        key=f"docid_{domain.value}",
    )

with col2:
    st.subheader("🖼️ Evidence Image")
    uploaded_file = st.file_uploader(
        "Upload supporting image evidence",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"upload_{domain.value}",
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded evidence", width='stretch')

st.markdown("---")
analyze_btn = st.button(
    "🔍 Analyze Evidence",
    type="primary",
    use_container_width=True
)

if analyze_btn:
    if not claim_text.strip():
        st.error("Please enter your claim text.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload a supporting image.")
        st.stop()

    try:
        text_client   = TextModelClient()
        vision_client = VisionModelClient()
    except ValueError as e:
        st.error(f"API key error: {e}")
        st.stop()

    with st.spinner("Analyzing evidence..."):
        import time
        start_time = time.time()
        try:
            # Merge evidence image with any images extracted from the document
            all_images = {}
            if uploaded_file:
                all_images["evidence_img"] = Image.open(uploaded_file)
            all_images.update(doc_images)  # add images extracted from PDF/DOCX

            if not all_images:
                st.error("Please provide at least one image as evidence.")
                st.stop()

            report = run_pipeline(
                text_client=text_client,
                vision_client=vision_client,
                document_id=document_id,
                domain=domain,
                document_text=claim_text,
                images=all_images,
            )
            st.session_state.report = report
            st.session_state.elapsed = time.time() - start_time
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Results — show if report exists in session state
# ---------------------------------------------------------------------------
if st.session_state.report:
    report = st.session_state.report
    st.success(f"Analysis complete — {len(report.claim_verdicts)} claims evaluated in {st.session_state.get('elapsed', 0):.1f} seconds.")
    # Show PII protection summary if any entities were masked
    if report.overall_risk_note:
        st.info(f"🔒 {report.overall_risk_note}")
    st.subheader("📋 Compliance Report")

    for i, cv in enumerate(report.claim_verdicts, 1):
        # Handle both dict and object (defensive)
        if isinstance(cv, dict):
            verdict_value = cv["verdict"]
            claim_txt     = cv["claim_text"]
            confidence    = cv["confidence"]
            explanation   = cv["explanation"]
        else:
            verdict_value = cv.verdict.value
            claim_txt     = cv.claim_text
            confidence    = cv.confidence
            explanation   = cv.explanation

        icon, color, _ = VERDICT_CONFIG.get(verdict_value, ("❓", "gray", ""))

        with st.expander(
            f"Claim {i} — {icon} {verdict_value} ({confidence:.0%} confidence)",
            expanded=True
        ):
            st.markdown(f"**Claim:** {claim_txt}")
            st.markdown(f"**Explanation:** {explanation}")

            col_v, col_c = st.columns([1, 1])
            with col_v:
                st.markdown(f"**Verdict:** {icon} `{verdict_value}`")
            with col_c:
                st.metric("Confidence", f"{confidence:.0%}")

    # ── Download options ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Report")
    col_dl1, col_dl2 = st.columns(2)

    report_json = json.dumps(report.model_dump(), indent=2, default=str)

    with col_dl1:
        st.download_button(
            label="📄 Download PDF Report",
            data=generate_pdf_report(
                report,
                elapsed_seconds=st.session_state.get("elapsed", 0)
            ),
            file_name=f"{document_id}_compliance_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            label="📊 Download JSON Report",
            data=report_json,
            file_name=f"{document_id}_compliance_report.json",
            mime="application/json",
            use_container_width=True,
        )