"""
Streamlit frontend for the Multimodal Lie-of-Omission Detector.
User inputs their claim text and uploads supporting image evidence.
"""

import streamlit as st
import json
import os
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import Domain
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Compliance Evidence Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Lie-of-Omission Detector")
st.markdown("Upload your claim and supporting evidence. The system will analyze whether your visual evidence supports, contradicts, or fails to substantiate your claims.")

# ---------------------------------------------------------------------------
# Sidebar — domain selection and instructions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    domain = st.selectbox(
        "Select claim domain",
        options=[
            Domain.HEALTH_INSURANCE,
            Domain.LOAN_APPLICATION,
            Domain.EVIDENCE_REVIEW,
            Domain.LICENSING_EMPLOYEE_VERIFICATION,
        ],
        format_func=lambda d: {
            Domain.HEALTH_INSURANCE: "Health Insurance",
            Domain.LOAN_APPLICATION: "Loan Application",
            Domain.EVIDENCE_REVIEW: "Evidence Review",
            Domain.LICENSING_EMPLOYEE_VERIFICATION: "Licensing & Employee Verification",
        }[d]
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Select your claim domain
    2. Type your claim in plain language
    3. Upload your supporting image
    4. Click **Analyze Evidence**
    5. Review the verdict for each claim
    """)

    st.markdown("---")
    st.markdown("### Verdict types")
    st.markdown("""
    - ✅ **Supported** — evidence confirms claim
    - 🔶 **Partially Supported** — evidence partially confirms
    - ❌ **Contradicted** — evidence contradicts claim
    - ⚠️ **Insufficient Evidence** — cannot determine
    - 🚫 **Missing Expected Evidence** — required region not visible
    """)

# ---------------------------------------------------------------------------
# Main area — user inputs
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Your Claim")
    claim_text = st.text_area(
        label="Describe your claim in plain language",
        placeholder="""Example:
My vehicle sustained severe damage to the front driver-side door during the accident.
The rear passenger door also has denting.
The windshield cracked due to the impact.""",
        height=250,
    )

    document_id = st.text_input(
        "Claim reference ID (optional)",
        value="claim_001",
        help="A unique identifier for this claim for your records"
    )

with col2:
    st.subheader("🖼️ Supporting Evidence")
    uploaded_file = st.file_uploader(
        "Upload your evidence image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a photo that serves as evidence for your claim"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded evidence", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
st.markdown("---")
analyze_btn = st.button("🔍 Analyze Evidence", type="primary", use_container_width=True)

if analyze_btn:
    # Input validation
    if not claim_text.strip():
        st.error("Please enter your claim text before analyzing.")
        st.stop()

    if not uploaded_file:
        st.error("Please upload a supporting image before analyzing.")
        st.stop()

    # Initialize clients
    try:
        text_client   = TextModelClient()
        vision_client = VisionModelClient()
    except ValueError as e:
        st.error(f"API key error: {e}")
        st.stop()

    # Run pipeline
    with st.spinner("Analyzing your evidence... this may take 15-30 seconds."):
        try:
            image = Image.open(uploaded_file)
            images = {"evidence_img": image}

            report = run_pipeline(
                text_client=text_client,
                vision_client=vision_client,
                document_id=document_id,
                domain=domain,
                document_text=claim_text,
                images=images,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ---------------------------------------------------------------------------
    # Results display
    # ---------------------------------------------------------------------------
    st.success(f"Analysis complete. {len(report.claim_verdicts)} claims evaluated.")
    st.subheader("📋 Compliance Report")

    # Verdict color mapping
    VERDICT_CONFIG = {
        "Supported":                  ("✅", "green"),
        "Partially Supported":        ("🔶", "orange"),
        "Contradicted":               ("❌", "red"),
        "Insufficient Evidence":      ("⚠️", "gray"),
        "Missing Expected Evidence":  ("🚫", "red"),
    }

    for i, cv in enumerate(report.claim_verdicts, 1):
        verdict_value = cv.verdict if isinstance(cv, dict) else cv.verdict.value
        claim_txt     = cv.claim_text if isinstance(cv, dict) else cv.claim_text
        confidence    = cv.confidence if isinstance(cv, dict) else cv.confidence
        explanation   = cv.explanation if isinstance(cv, dict) else cv.explanation

        icon, color = VERDICT_CONFIG.get(verdict_value, ("❓", "gray"))

        with st.expander(f"Claim {i}: {claim_txt[:80]}...", expanded=True):
            col_a, col_b = st.columns([2, 1])

            with col_a:
                st.markdown(f"**Full claim:** {claim_txt}")
                st.markdown(f"**Explanation:** {explanation}")

            with col_b:
                st.markdown(f"**Verdict:** {icon} `{verdict_value}`")
                st.metric("Confidence", f"{confidence:.0%}")

    # ---------------------------------------------------------------------------
    # Download report as JSON
    # ---------------------------------------------------------------------------
    report_dict = report.model_dump()
    report_json = json.dumps(report_dict, indent=2, default=str)

    st.download_button(
        label="⬇️ Download Full Report (JSON)",
        data=report_json,
        file_name=f"{document_id}_compliance_report.json",
        mime="application/json",
    )