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
from src.eval.evaluator import run_evaluation, format_report
from src.reporting.report_generator import generate_pdf_report, generate_combined_pdf_report
from src.claim_generation.claim_generator import prefill_from_evidence, generate_claim_document
from src.claim_generation.form_schema import CAR_INSURANCE_CLAIM_SCHEMA, SECTIONS, get_fields_for_section
import copy

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
    "selected_domain":  Domain.VEHICLE_INSURANCE,
    "report":           None,
    "elapsed":          0,
    "doc_bytes":        None,
    "doc_name":         None,
    "doc_claim_text":   "",
    "doc_images":       {},
    "doc_notes":        [],
    "doc_page_count":   0,
    "img_bytes":        None,
    "img_name":         None,
    "manual_claim":     "",
    "additional_docs":  {},
    
    "wizard_schema":           None,
    "wizard_step":             0,
    "wizard_complete":         False,
    "wizard_observations":     "",
    "wizard_img_bytes":        None,
    "wizard_img_name":         None,
    "generated_claim_doc":     "",
    "gen_validation_report":   None,
    "gen_validation_elapsed":  0,
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

    # FIX 1 — preserve uploaded files across domain switch
    if selected != st.session_state.selected_domain:
        saved_img_bytes  = st.session_state.img_bytes
        saved_img_name   = st.session_state.img_name
        saved_doc_bytes  = st.session_state.doc_bytes
        saved_doc_name   = st.session_state.doc_name
        saved_doc_text   = st.session_state.doc_claim_text
        saved_doc_images = st.session_state.doc_images
        saved_doc_notes  = st.session_state.doc_notes
        saved_doc_pages  = st.session_state.doc_page_count
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.selected_domain  = selected
        st.session_state.img_bytes        = saved_img_bytes
        st.session_state.img_name         = saved_img_name
        st.session_state.doc_bytes        = saved_doc_bytes
        st.session_state.doc_name         = saved_doc_name
        st.session_state.doc_claim_text   = saved_doc_text
        st.session_state.doc_images       = saved_doc_images
        st.session_state.doc_notes        = saved_doc_notes
        st.session_state.doc_page_count   = saved_doc_pages
        st.rerun()

    domain = st.session_state.selected_domain

    st.markdown("---")
    st.markdown("### Verdict guide")
    for verdict, (icon, _, desc) in VERDICT_CONFIG.items():
        st.markdown(f"{icon} **{verdict}** — {desc}")

# ── Main title ────────────────────────────────────────────────
st.title("🔍 Compliance Evidence Analyzer")

main_tab, generate_tab, eval_tab = st.tabs([
    "📋 Analyze Claim",
    "✍️ Generate Claim",
    "📊 Evaluation Dashboard"
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE CLAIM
# ═════════════════════════════════════════════════════════════
with main_tab:
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

    # ── Left column ───────────────────────────────────────────
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
            if claim_text_input:
                st.session_state.manual_claim = claim_text_input

            # FIX 3 — removed the four lines that cleared doc state here
            # They were wiping uploaded files on every rerun

            claim_text = st.session_state.manual_claim

        else:
            st.subheader("📂 Upload Document")
            doc_file = st.file_uploader(
                "Upload your claim document",
                type=["pdf", "docx", "txt"],
                key=f"doc_{domain.value}",
            )

            if doc_file is not None:
                if doc_file.name != st.session_state.doc_name:
                    raw_bytes = doc_file.read()
                    from src.ingestion.document_loader import load_document
                    with st.spinner("Extracting document content..."):
                        content = load_document(raw_bytes, doc_file.name)

                    st.session_state.doc_bytes      = raw_bytes
                    st.session_state.doc_name       = doc_file.name
                    st.session_state.doc_claim_text = content.extracted_text
                    st.session_state.doc_images     = content.extracted_images
                    st.session_state.doc_notes      = content.extraction_notes
                    st.session_state.doc_page_count = content.page_count

                    # FIX 2 — show suggestion only, never auto-rerun on classification
                    if content.extracted_text:
                        try:
                            text_client_tmp = TextModelClient()
                            classification  = classify_document(
                                text_client_tmp, content.extracted_text
                            )
                            if classification.confidence > 0.6:
                                suggested = classification.suggested_domain
                                if suggested != st.session_state.selected_domain:
                                    st.warning(
                                        f"💡 Detected: "
                                        f"**{classification.document_type.replace('_', ' ').title()}**. "
                                        f"Suggested domain: "
                                        f"**{suggested.value.replace('_', ' ').title()}**. "
                                        f"Switch in the sidebar if needed."
                                    )
                                else:
                                    st.success(
                                        f"📋 Document classified as: "
                                        f"**{classification.document_type.replace('_', ' ').title()}**"
                                    )
                            else:
                                st.warning(
                                    "Document type unclear. "
                                    "Please select the correct domain from the sidebar."
                                )
                        except Exception:
                            pass

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

        # Multi-document upload
        st.markdown("---")
        st.markdown("**Additional Documents** *(optional — for cross-document analysis)*")
        st.caption(
            "Upload additional documents to check for contradictions across them."
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
                    st.session_state.doc_images.update(content.extracted_images)
            st.session_state.additional_docs = additional_docs
            if additional_docs:
                st.success(
                    f"{len(additional_docs)} additional document(s) loaded "
                    f"for cross-document analysis."
                )
        else:
            st.session_state.additional_docs = {}

    # ── Right column ──────────────────────────────────────────
    with col2:
        st.subheader("🖼️ Evidence Image")
        uploaded_file = st.file_uploader(
            "Upload supporting image evidence",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"upload_{domain.value}",
        )

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.img_name:
                st.session_state.img_bytes = uploaded_file.read()
                st.session_state.img_name  = uploaded_file.name

        if st.session_state.img_bytes:
            img_display = Image.open(io.BytesIO(st.session_state.img_bytes))
            st.image(
                img_display,
                caption=f"Evidence: {st.session_state.img_name}",
                width='stretch'
            )

    # ── Analyse button ────────────────────────────────────────
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
                if st.session_state.img_bytes:
                    all_images["evidence_img"] = Image.open(
                        io.BytesIO(st.session_state.img_bytes)
                    )
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

    # ── Results ───────────────────────────────────────────────
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

        # ── Downloads ─────────────────────────────────────────
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

# ═════════════════════════════════════════════════════════════
# TAB 2 — GENERATE CLAIM
# ═════════════════════════════════════════════════════════════
with generate_tab:
    st.markdown("## ✍️ AI-Powered Claim Generation")
    st.markdown(
        "Upload your evidence image. The AI will analyze it, "
        "pre-fill your claim form, and generate a formal claim document. "
        "You can then validate it and download a combined report."
    )
    st.markdown("---")

    gen_uploaded = st.file_uploader(
        "Upload your evidence image to begin",
        type=["jpg", "jpeg", "png", "webp"],
        key="gen_evidence_upload",
    )

    if gen_uploaded is not None:
        if gen_uploaded.name != st.session_state.wizard_img_name:
            st.session_state.wizard_img_bytes    = gen_uploaded.read()
            st.session_state.wizard_img_name     = gen_uploaded.name
            st.session_state.wizard_schema       = None
            st.session_state.wizard_step         = 0
            st.session_state.wizard_complete     = False
            st.session_state.generated_claim_doc = ""
            if "gen_validation_report" in st.session_state:
                st.session_state.gen_validation_report = None

    if st.session_state.wizard_img_bytes:
        gen_img = Image.open(io.BytesIO(st.session_state.wizard_img_bytes))
        st.image(gen_img, caption="Evidence image", width='stretch')

        # ── AI pre-fill ───────────────────────────────────────
        if st.session_state.wizard_schema is None:
            if st.button(
                "🤖 Analyze Evidence & Start Claim Form",
                type="primary",
                use_container_width=True,
            ):
                try:
                    vision_client_g = VisionModelClient()
                except ValueError as e:
                    st.error(f"API key error: {e}")
                    st.stop()

                with st.spinner("Analyzing evidence — pre-filling claim form..."):
                    try:
                        schema, observations = prefill_from_evidence(
                            vision_client_g, gen_img
                        )
                        st.session_state.wizard_schema      = schema
                        st.session_state.wizard_observations = observations
                        st.session_state.wizard_step        = 0
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.stop()

        # ── Wizard ────────────────────────────────────────────
        if st.session_state.wizard_schema is not None and not st.session_state.wizard_complete:
            schema = st.session_state.wizard_schema
            step   = st.session_state.wizard_step

            if st.session_state.wizard_observations:
                with st.expander("🔍 AI observations from your evidence", expanded=True):
                    st.info(st.session_state.wizard_observations)

            st.progress(
                step / len(SECTIONS),
                text=f"Step {step + 1} of {len(SECTIONS)}: {SECTIONS[step]}"
            )
            st.subheader(f"📋 {SECTIONS[step]}")

            if SECTIONS[step] == "Policyholder Information":
                st.caption(
                    "ℹ️ The AI cannot determine your personal information. "
                    "Please fill these fields manually."
                )
            else:
                st.caption(
                    "✨ Fields marked with *(AI pre-filled)* were detected from your evidence. "
                    "Please review and edit if needed."
                )

            section_fields = get_fields_for_section(SECTIONS[step])
            updated_fields = {}

            # Track police report selection for conditional field
            police_filed_val = ""

            for field in section_fields:
                current_val = field.value or ""
                label = field.label
                if field.ai_filled:
                    label = f"{field.label} *(AI pre-filled — please confirm)*"

                # Conditional: hide police report number if "No" selected
                if field.key == "police_report_number":
                    if police_filed_val == "No":
                        updated_fields[field.key] = ""
                        continue

                if field.field_type == "textarea":
                    val = st.text_area(
                        label,
                        value=current_val,
                        height=150,
                        key=f"wiz_{step}_{field.key}",
                        placeholder=field.placeholder,
                    )
                elif field.field_type == "yes_no":
                    options = ["", "Yes", "No"]
                    idx = options.index(current_val) if current_val in options else 0
                    val = st.selectbox(
                        label,
                        options=options,
                        index=idx,
                        key=f"wiz_{step}_{field.key}",
                    )
                    if field.key == "police_report_filed":
                        police_filed_val = val
                elif field.field_type == "date":
                    val = st.text_input(
                        label,
                        value=current_val,
                        key=f"wiz_{step}_{field.key}",
                        placeholder="DD/MM/YYYY",
                    )
                elif field.field_type == "number":
                    val = st.text_input(
                        label,
                        value=current_val,
                        key=f"wiz_{step}_{field.key}",
                        placeholder=field.placeholder,
                    )
                else:
                    val = st.text_input(
                        label,
                        value=current_val,
                        key=f"wiz_{step}_{field.key}",
                        placeholder=field.placeholder,
                    )

                updated_fields[field.key] = val

            # Save user input to schema
            for f in schema:
                if f.key in updated_fields:
                    f.value = updated_fields[f.key]
                    if updated_fields[f.key]:
                        f.user_confirmed = True

            st.markdown("---")
            nav1, nav2 = st.columns(2)
            with nav1:
                if step > 0:
                    if st.button("← Previous", use_container_width=True):
                        st.session_state.wizard_step -= 1
                        st.rerun()
            with nav2:
                if step < len(SECTIONS) - 1:
                    if st.button("Next →", type="primary", use_container_width=True):
                        st.session_state.wizard_step += 1
                        st.rerun()
                else:
                    if st.button(
                        "✅ Generate Claim Document",
                        type="primary",
                        use_container_width=True
                    ):
                        st.session_state.wizard_complete = True
                        st.rerun()

        # ── Claim document + inline validation ────────────────
        if st.session_state.wizard_complete:

            if not st.session_state.generated_claim_doc:
                try:
                    text_client_g = TextModelClient()
                except ValueError as e:
                    st.error(f"API key error: {e}")
                    st.stop()

                with st.spinner("Writing your formal claim document..."):
                    try:
                        claim_doc = generate_claim_document(
                            text_client_g,
                            st.session_state.wizard_schema
                        )
                        st.session_state.generated_claim_doc = claim_doc
                    except Exception as e:
                        st.error(f"Claim generation failed: {e}")
                        st.stop()

            if st.session_state.generated_claim_doc:
                st.subheader("📄 Your Generated Claim Document")
                edited_claim = st.text_area(
                    "Review and edit your claim if needed",
                    value=st.session_state.generated_claim_doc,
                    height=350,
                    key="final_claim_edit",
                )

                st.markdown("---")
                st.subheader("🔍 Validate Claim Against Evidence")
                st.markdown(
                    "Click below to validate your claim against the uploaded evidence. "
                    "The system will check every claim point and generate a combined PDF."
                )

                if st.button(
                    "🔍 Validate & Generate Combined Report",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        text_client_v   = TextModelClient()
                        vision_client_v = VisionModelClient()
                    except ValueError as e:
                        st.error(f"API key error: {e}")
                        st.stop()

                    with st.spinner("Validating claim against evidence..."):
                        start_v = time.time()
                        try:
                            val_images = {
                                "evidence_img": Image.open(
                                    io.BytesIO(st.session_state.wizard_img_bytes)
                                )
                            }
                            val_report = run_pipeline(
                                text_client=text_client_v,
                                vision_client=vision_client_v,
                                document_id="generated_claim",
                                domain=st.session_state.selected_domain,
                                document_text=edited_claim,
                                images=val_images,
                            )
                            elapsed_v = time.time() - start_v
                            st.session_state.gen_validation_report = val_report
                            st.session_state.gen_validation_elapsed = elapsed_v
                        except Exception as e:
                            st.error(f"Validation failed: {e}")
                            st.stop()

                # ── Show validation results inline ────────────
                if st.session_state.get("gen_validation_report"):
                    val_report = st.session_state.gen_validation_report
                    elapsed_v  = st.session_state.get("gen_validation_elapsed", 0)

                    st.success(
                        f"Validation complete — {len(val_report.claim_verdicts)} "
                        f"claims checked in {elapsed_v:.1f} seconds."
                    )

                    if val_report.overall_risk_note:
                        st.info(f"🔒 {val_report.overall_risk_note}")

                    st.subheader("📋 Validation Results")
                    for i, cv in enumerate(val_report.claim_verdicts, 1):
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

                        icon, color, _ = VERDICT_CONFIG.get(
                            verdict_value, ("❓", "gray", "")
                        )
                        with st.expander(
                            f"Claim {i} — {icon} {verdict_value} ({confidence:.0%})",
                            expanded=True
                        ):
                            st.markdown(f"**Claim:** {claim_txt}")
                            st.markdown(f"**Explanation:** {explanation}")
                            col_vv, col_cc = st.columns(2)
                            with col_vv:
                                st.markdown(f"**Verdict:** {icon} `{verdict_value}`")
                            with col_cc:
                                st.metric("Confidence", f"{confidence:.0%}")

                    # ── Combined PDF download ─────────────────
                    st.markdown("---")
                    st.subheader("⬇️ Download Combined Report")
                    try:
                        combined_pdf = generate_combined_pdf_report(
                            claim_document=edited_claim,
                            report=val_report,
                            elapsed_seconds=elapsed_v,
                        )
                        st.download_button(
                            label="📄 Download Combined Claim + Validation PDF",
                            data=combined_pdf,
                            file_name="claim_and_validation_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

                # ── Start over ────────────────────────────────
                st.markdown("---")
                if st.button("🔄 Start Over", use_container_width=True):
                    for k in [
                        "wizard_schema", "wizard_step", "wizard_complete",
                        "wizard_observations", "wizard_img_bytes", "wizard_img_name",
                        "generated_claim_doc", "gen_validation_report",
                    ]:
                        st.session_state[k] = None if "schema" in k or "report" in k else (
                            0 if "step" in k else (
                                False if "complete" in k else (
                                    "" if "bytes" not in k else None
                                )
                            )
                        )
                    st.rerun()

# ═════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION DASHBOARD
# ═════════════════════════════════════════════════════════════
with eval_tab:
    st.subheader("📊 Gold Evaluation Set — Accuracy Measurement")
    st.markdown(
        "Run the pipeline against hand-labeled gold cases to measure "
        "accuracy, precision, and recall across all domains."
    )
    st.info(
        "Before running: place your test images in "
        "`data/sample_images/` with filenames matching `cases.json`."
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        eval_domain = st.selectbox(
            "Filter by domain (optional)",
            options=[
                "All domains",
                "vehicle_insurance",
                "health_insurance",
                "loan_application",
                "evidence_review",
                "licensing_employee_verification",
            ],
            key="eval_domain"
        )
    with col_e2:
        max_cases = st.number_input(
            "Max cases to run",
            min_value=1,
            max_value=50,
            value=5,
            key="eval_max"
        )
    with col_e3:
        st.markdown("&nbsp;")
        run_eval_btn = st.button(
            "▶ Run Evaluation",
            type="primary",
            use_container_width=True
        )

    if run_eval_btn:
        try:
            text_client_e   = TextModelClient()
            vision_client_e = VisionModelClient()
        except ValueError as e:
            st.error(f"API key error: {e}")
            st.stop()

        with st.spinner(f"Running evaluation on up to {max_cases} cases..."):
            eval_report = run_evaluation(
                gold_cases_path="data/gold_eval/cases.json",
                images_dir="data/sample_images",
                text_client=text_client_e,
                vision_client=vision_client_e,
                max_cases=int(max_cases),
                domain_filter=(
                    None if eval_domain == "All domains" else eval_domain
                ),
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{eval_report.overall_accuracy:.1%}")
        m2.metric("Claims Evaluated", str(eval_report.total_claims))
        m3.metric("Correct",          str(eval_report.correct_claims))
        m4.metric("Avg Time/Case",    f"{eval_report.avg_elapsed_seconds:.1f}s")

        st.markdown("### Domain Accuracy")
        for dom, acc in eval_report.domain_accuracy.items():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(acc, text=dom.replace("_", " ").title())
            with col_b:
                st.markdown(f"**{acc:.1%}**")

        st.markdown("### Verdict Precision / Recall")
        table_data = {
            "Verdict Type": list(eval_report.verdict_precision.keys()),
            "Precision": [
                f"{v:.1%}"
                for v in eval_report.verdict_precision.values()
            ],
            "Recall": [
                f"{v:.1%}"
                for v in eval_report.verdict_recall.values()
            ],
        }
        st.table(table_data)

        st.markdown("### Confidence Calibration")
        cal1, cal2 = st.columns(2)
        cal1.metric(
            "Avg Confidence (Correct)",
            f"{eval_report.avg_confidence_correct:.1%}",
        )
        cal2.metric(
            "Avg Confidence (Incorrect)",
            f"{eval_report.avg_confidence_incorrect:.1%}",
        )

        if eval_report.failure_analysis:
            st.markdown("### Failure Analysis")
            for failure in eval_report.failure_analysis:
                st.error(failure)

        eval_dict = {
            "overall_accuracy":         eval_report.overall_accuracy,
            "total_cases":              eval_report.total_cases,
            "total_claims":             eval_report.total_claims,
            "correct_claims":           eval_report.correct_claims,
            "domain_accuracy":          eval_report.domain_accuracy,
            "verdict_precision":        eval_report.verdict_precision,
            "verdict_recall":           eval_report.verdict_recall,
            "avg_confidence_correct":   eval_report.avg_confidence_correct,
            "avg_confidence_incorrect": eval_report.avg_confidence_incorrect,
            "failure_analysis":         eval_report.failure_analysis,
        }
        st.download_button(
            label="⬇️ Download Eval Results (JSON)",
            data=json.dumps(eval_dict, indent=2),
            file_name="eval_results.json",
            mime="application/json",
        )