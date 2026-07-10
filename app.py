from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import os
import io
import time
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import Domain
from src.pipeline import run_pipeline
from src.reporting.report_generator import generate_pdf_report
from src.classification.document_classifier import classify_document

st.set_page_config(
    page_title="Compliance Evidence Analyzer",
    page_icon="🔍",
    layout="wide"
)

DOMAIN_LABELS = {
    Domain.VEHICLE_INSURANCE:               "🚗 Vehicle Insurance",
    Domain.HEALTH_INSURANCE:                "🏥 Health Insurance",
    Domain.LOAN_APPLICATION:                "💰 Loan Application",
    Domain.EVIDENCE_REVIEW:                 "⚖️ Evidence Review",
    Domain.LICENSING_EMPLOYEE_VERIFICATION: "🪪 Licensing & Employee Verification",
}

VERDICT_CONFIG = {
    "Supported":                ("✅", "green",  "Evidence clearly confirms this claim."),
    "Partially Supported":      ("🔶", "orange", "Evidence partially confirms this claim."),
    "Contradicted":             ("❌", "red",    "Evidence directly contradicts this claim."),
    "Insufficient Evidence":    ("⚠️", "gray",   "Evidence is unclear or ambiguous."),
    "Missing Expected Evidence":("🚫", "red",    "Required region not visible in the image."),
}

# ── Session state initialisation ──────────────────────────────
defaults = {
    "selected_domain":    Domain.VEHICLE_INSURANCE,
    "report":             None,
    "elapsed":            0,
    "doc_bytes":          None,
    "doc_name":           None,
    "doc_claim_text":     "",
    "doc_images":         {},
    "doc_notes":          [],
    "doc_page_count":     0,
    "img_bytes":          None,
    "img_name":           None,
    "manual_claim":       "",
    "additional_docs":    {},   # label -> text for multi-doc mode
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    selected = st.selectbox(
        "Claim domain",
        options=list(DOMAIN_LABELS.keys()),
        format_func=lambda d: DOMAIN_LABELS[d],
        index=0,
    )

    # Domain changed — clear all file state
    if selected != st.session_state.selected_domain:
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.selected_domain = selected
        st.rerun()

    domain = st.session_state.selected_domain

    st.markdown("---")
    st.markdown("### Verdict guide")
    for verdict, (icon, _, desc) in VERDICT_CONFIG.items():
        st.markdown(f"{icon} **{verdict}** — {desc}")

# ── Main ──────────────────────────────────────────────────────
st.title("🔍 Compliance Evidence Analyzer")
st.markdown(f"**Domain:** {DOMAIN_LABELS[domain]}")
st.markdown("---")

input_mode = st.radio(
    "How do you want to provide your claim?",
    options=["Type claim manually", "Upload a document (PDF, DOCX, TXT)"],
    horizontal=True,
    key=f"input_mode_{domain.value}",
)

st.markdown("---")
col1, col2 = st.columns([1, 1])

# ── Left column ───────────────────────────────────────────────
with col1:
    if input_mode == "Type claim manually":
        st.subheader("📄 Your Claim")
        claim_text_input = st.text_area(
            label="Describe your claim in plain language",
            placeholder=(
                "Example: My car's left front door is severely damaged. "
                "The rear bumper has minor scratches. The windshield is intact."
            ),
            height=200,
            key=f"claim_{domain.value}",
        )
        # Persist manual claim text in session state
        if claim_text_input:
            st.session_state.manual_claim = claim_text_input

        # Clear doc state if mode switched
        st.session_state.doc_bytes      = None
        st.session_state.doc_name       = None
        st.session_state.doc_claim_text = ""
        st.session_state.doc_images     = {}

        claim_text = st.session_state.manual_claim

    else:
        st.subheader("📂 Upload Document")

        doc_file = st.file_uploader(
            "Upload your claim document",
            type=["pdf", "docx", "txt"],
            key=f"doc_{domain.value}",
        )

        # If a new file was just uploaded, process and save to session state
        if doc_file is not None:
            if doc_file.name != st.session_state.doc_name:
                raw_bytes = doc_file.read()
                from src.ingestion.document_loader import load_document
                with st.spinner("Extracting document content..."):
                    content = load_document(raw_bytes, doc_file.name)

                # Save everything to session state
                st.session_state.doc_bytes      = raw_bytes
                st.session_state.doc_name       = doc_file.name
                st.session_state.doc_claim_text = content.extracted_text
                st.session_state.doc_images     = content.extracted_images
                st.session_state.doc_notes      = content.extraction_notes
                st.session_state.doc_page_count = content.page_count

                # Auto-classify
                if content.extracted_text:
                    try:
                        text_client_tmp = TextModelClient()
                        classification  = classify_document(
                            text_client_tmp, content.extracted_text
                        )
                        if (
                            classification.confidence > 0.6
                            and classification.suggested_domain
                            != st.session_state.selected_domain
                        ):
                            st.session_state.selected_domain = (
                                classification.suggested_domain
                            )
                            st.rerun()
                    except Exception:
                        pass

        # Always show status from session state (persists across reruns)
        if st.session_state.doc_name:
            st.success(
                f"Document loaded: **{st.session_state.doc_name}** "
                f"({st.session_state.doc_page_count} page(s), "
                f"{len(st.session_state.doc_images)} embedded image(s))"
            )
            for note in st.session_state.doc_notes:
                st.info(note)
            if st.session_state.doc_claim_text:
                with st.expander("Preview extracted text", expanded=False):
                    preview = st.session_state.doc_claim_text
                    st.text(preview[:2000] + ("..." if len(preview) > 2000 else ""))

        claim_text = st.session_state.doc_claim_text
        st.session_state.manual_claim = ""

    document_id = st.text_input(
        "Reference ID",
        value="claim_001",
        key=f"docid_{domain.value}",
    )

    # Multi-document upload — Phase 5
    st.markdown("---")
    st.markdown("**Additional Documents** *(optional — for cross-document analysis)*")
    st.caption(
        "Upload additional documents to check for contradictions across them. "
        "Example: upload a court ruling AND a police report to detect factual conflicts."
    )

    extra_files = st.file_uploader(
        "Upload additional documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"extra_docs_{domain.value}",
    )

    if extra_files:
        from src.ingestion.document_loader import load_document
        additional_docs = {}
        for ef in extra_files:
            raw = ef.read()
            content = load_document(raw, ef.name)
            if content.extracted_text:
                label = ef.name.replace(" ", "_").replace(".", "_")
                additional_docs[label] = content.extracted_text
                # merge any images from additional docs
                st.session_state.doc_images.update(content.extracted_images)
        st.session_state.additional_docs = additional_docs
        if additional_docs:
            st.success(
                f"{len(additional_docs)} additional document(s) loaded for "
                f"cross-document analysis."
            )
    else:
        st.session_state.additional_docs = {}

# ── Right column ──────────────────────────────────────────────
with col2:
    st.subheader("🖼️ Evidence Image")

    uploaded_file = st.file_uploader(
        "Upload supporting image evidence",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"upload_{domain.value}",
    )

    # If a new image was uploaded, save bytes to session state
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.img_name:
            st.session_state.img_bytes = uploaded_file.read()
            st.session_state.img_name  = uploaded_file.name

    # Always display from session state
    if st.session_state.img_bytes:
        img_display = Image.open(io.BytesIO(st.session_state.img_bytes))
        st.image(
            img_display,
            caption=f"Evidence: {st.session_state.img_name}",
            use_container_width=True
        )

# ── Analyse button ────────────────────────────────────────────
st.markdown("---")
analyze_btn = st.button(
    "🔍 Analyze Evidence",
    type="primary",
    use_container_width=True
)

if analyze_btn:
    if not claim_text or not claim_text.strip():
        st.error("Please provide your claim (type it or upload a document).")
        st.stop()
    if not st.session_state.img_bytes and not st.session_state.doc_images:
        st.error("Please upload at least one image as evidence.")
        st.stop()

    try:
        text_client   = TextModelClient()
        vision_client = VisionModelClient()
    except ValueError as e:
        st.error(f"API key error: {e}")
        st.stop()

    with st.spinner("Analyzing evidence..."):
        start_time = time.time()
        try:
            all_images = {}
            # Primary evidence image (from image uploader)
            if st.session_state.img_bytes:
                all_images["evidence_img"] = Image.open(
                    io.BytesIO(st.session_state.img_bytes)
                )
            # Images extracted from uploaded document
            all_images.update(st.session_state.doc_images)

            report = run_pipeline(
                text_client=text_client,
                vision_client=vision_client,
                document_id=document_id,
                domain=domain,
                document_text=claim_text,
                images=all_images,
                additional_documents=st.session_state.additional_docs or None,
            )
            st.session_state.report  = report
            st.session_state.elapsed = time.time() - start_time
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# ── Results ───────────────────────────────────────────────────
if st.session_state.report:
    report  = st.session_state.report
    elapsed = st.session_state.get("elapsed", 0)

    st.success(
        f"Analysis complete — {len(report.claim_verdicts)} claims "
        f"evaluated in {elapsed:.1f} seconds."
    )

    if report.overall_risk_note:
        note = report.overall_risk_note
        if "CROSS-DOCUMENT ANALYSIS" in note:
            parts = note.split(" | ")
            for part in parts:
                if "CROSS-DOCUMENT" in part:
                    if "contradiction(s) found" in part:
                        st.error(f"⚔️ {part}")
                    else:
                        st.success(f"✅ {part}")
                else:
                    st.info(f"🔒 {part}")
        else:
            st.info(f"🔒 {note}")

    st.subheader("📋 Compliance Report")

    for i, cv in enumerate(report.claim_verdicts, 1):
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
                st.metric("Substantiation Confidence", f"{confidence:.0%}")

    # ── Downloads ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Report")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        try:
            pdf_bytes = generate_pdf_report(report, elapsed_seconds=elapsed)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{document_id}_compliance_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    with col_dl2:
        report_json = json.dumps(report.model_dump(), indent=2, default=str)
        st.download_button(
            label="📊 Download JSON Report",
            data=report_json,
            file_name=f"{document_id}_compliance_report.json",
            mime="application/json",
            use_container_width=True,
        )