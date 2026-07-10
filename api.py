"""
FastAPI Backend — Phase 7.
Wraps the pipeline in REST endpoints so any frontend can call it.

Endpoints:
  GET  /health              — system health check
  GET  /domains             — list available domains
  POST /analyze             — main pipeline endpoint
  POST /analyze/multi       — multi-document analysis
  GET  /report/{report_id}  — retrieve stored report
"""

from dotenv import load_dotenv
load_dotenv()

import io
import json
import uuid
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import Domain
from src.pipeline import run_pipeline
from src.ingestion.document_loader import load_document
from src.pii.detector import PIIRegistry

app = FastAPI(
    title="Multimodal Lie-of-Omission Detector API",
    description=(
        "AI-powered compliance assistant that detects lies of omission "
        "in insurance claims, loan applications, evidence reviews, "
        "and employee verification documents."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory report store — replace with DB in production
_report_store: dict[str, dict] = {}

DOMAIN_MAP = {
    "vehicle_insurance":               Domain.VEHICLE_INSURANCE,
    "health_insurance":                Domain.HEALTH_INSURANCE,
    "loan_application":                Domain.LOAN_APPLICATION,
    "evidence_review":                 Domain.EVIDENCE_REVIEW,
    "licensing_employee_verification": Domain.LICENSING_EMPLOYEE_VERIFICATION,
}


def _get_clients():
    try:
        return TextModelClient(), VisionModelClient()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"API key error: {e}")


def _parse_domain(domain_str: str) -> Domain:
    domain = DOMAIN_MAP.get(domain_str.lower())
    if not domain:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid domain '{domain_str}'. "
                f"Valid options: {list(DOMAIN_MAP.keys())}"
            )
        )
    return domain


# ── GET /health ───────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "version":   "1.0.0",
        "models": {
            "text":   "llama-3.3-70b-versatile (Groq)",
            "vision": "meta-llama/llama-4-scout-17b-16e-instruct (Groq)",
        }
    }


# ── GET /domains ──────────────────────────────────────────────
@app.get("/domains")
async def list_domains():
    return {
        "domains": [
            {
                "id":          k,
                "label":       k.replace("_", " ").title(),
                "description": {
                    "vehicle_insurance":               "Vehicle accident and damage claims",
                    "health_insurance":                "Medical treatment and injury claims",
                    "loan_application":                "Income, employment and identity verification",
                    "evidence_review":                 "Court rulings and legal document analysis",
                    "licensing_employee_verification": "License validity and credential checks",
                }[k]
            }
            for k in DOMAIN_MAP
        ]
    }


# ── POST /analyze ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    domain:       str         = Form(...),
    document_id:  str         = Form(default=""),
    claim_text:   str         = Form(default=""),
    document:     Optional[UploadFile] = File(default=None),
    image:        UploadFile  = File(...),
):
    """
    Main analysis endpoint.
    Provide either claim_text OR a document file (or both).
    Image is required as visual evidence.
    """
    parsed_domain = _parse_domain(domain)
    doc_id = document_id or str(uuid.uuid4())[:8]

    # ── Extract text ──────────────────────────────────────────
    final_claim_text = claim_text.strip()
    doc_images: dict = {}

    if document:
        doc_bytes = await document.read()
        content   = load_document(doc_bytes, document.filename)
        if content.extracted_text:
            final_claim_text = content.extracted_text
        doc_images = content.extracted_images

    if not final_claim_text:
        raise HTTPException(
            status_code=400,
            detail="Provide either claim_text or a document file."
        )

    # ── Load evidence image ───────────────────────────────────
    img_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open uploaded image. Use JPG, PNG, or WEBP."
        )

    all_images = {"evidence_img": pil_image}
    all_images.update(doc_images)

    # ── Run pipeline ──────────────────────────────────────────
    text_client, vision_client = _get_clients()

    try:
        report = run_pipeline(
            text_client=text_client,
            vision_client=vision_client,
            document_id=doc_id,
            domain=parsed_domain,
            document_text=final_claim_text,
            images=all_images,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    report_dict = report.model_dump()
    report_dict["report_id"]  = doc_id
    report_dict["created_at"] = datetime.now().isoformat()

    # Store for retrieval
    _report_store[doc_id] = report_dict

    return JSONResponse(content=report_dict)


# ── POST /analyze/multi ───────────────────────────────────────
@app.post("/analyze/multi")
async def analyze_multi(
    domain:      str        = Form(...),
    document_id: str        = Form(default=""),
    claim_text:  str        = Form(default=""),
    primary_doc: Optional[UploadFile] = File(default=None),
    image:       UploadFile = File(...),
    extra_docs:  list[UploadFile] = File(default=[]),
):
    """
    Multi-document analysis with cross-document contradiction detection.
    Upload a primary document + additional documents for comparison.
    """
    parsed_domain = _parse_domain(domain)
    doc_id = document_id or str(uuid.uuid4())[:8]

    final_claim_text = claim_text.strip()
    doc_images: dict = {}

    if primary_doc:
        doc_bytes = await primary_doc.read()
        content   = load_document(doc_bytes, primary_doc.filename)
        if content.extracted_text:
            final_claim_text = content.extracted_text
        doc_images = content.extracted_images

    if not final_claim_text:
        raise HTTPException(
            status_code=400,
            detail="Provide either claim_text or a primary document."
        )

    # Load additional documents
    additional_documents = {}
    for ef in extra_docs:
        ef_bytes = await ef.read()
        ef_content = load_document(ef_bytes, ef.filename)
        if ef_content.extracted_text:
            label = ef.filename.replace(" ", "_").replace(".", "_")
            additional_documents[label] = ef_content.extracted_text
            doc_images.update(ef_content.extracted_images)

    img_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    all_images = {"evidence_img": pil_image}
    all_images.update(doc_images)

    text_client, vision_client = _get_clients()

    try:
        report = run_pipeline(
            text_client=text_client,
            vision_client=vision_client,
            document_id=doc_id,
            domain=parsed_domain,
            document_text=final_claim_text,
            images=all_images,
            additional_documents=additional_documents or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    report_dict = report.model_dump()
    report_dict["report_id"]  = doc_id
    report_dict["created_at"] = datetime.now().isoformat()
    _report_store[doc_id] = report_dict

    return JSONResponse(content=report_dict)


# ── GET /report/{report_id} ───────────────────────────────────
@app.get("/report/{report_id}")
async def get_report(report_id: str):
    report = _report_store.get(report_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found."
        )
    return JSONResponse(content=report)


# ── GET /reports ──────────────────────────────────────────────
@app.get("/reports")
async def list_reports():
    return {
        "total": len(_report_store),
        "reports": [
            {
                "report_id":  rid,
                "domain":     r.get("domain"),
                "claims":     len(r.get("claim_verdicts", [])),
                "created_at": r.get("created_at"),
            }
            for rid, r in _report_store.items()
        ]
    }