# Multimodal Lie-of-Omission Detector for Compliance Documents

**Genesys Research Lab — Internship Project**

An AI-powered compliance assistant that jointly reasons over structured 
documents and visual evidence to detect lies of omission — cases where 
submitted evidence is technically authentic but fails to substantiate 
the written claims.

---

## What It Does

Traditional compliance systems treat text and images as independent 
inputs. This system reasons over them together, detecting:

- **Supported** — evidence clearly confirms the claim
- **Partially Supported** — evidence partially confirms
- **Contradicted** — evidence directly contradicts the claim
- **Insufficient Evidence** — image is ambiguous or unclear
- **Missing Expected Evidence** — required region not visible in image

---

## Scoped Domains

| Domain | Example Use Case |
|--------|------------------|
| Vehicle Insurance | Front door damage claim vs. car photo |
| Health Insurance  | Injury claim vs. medical photograph |
| Loan Application  | Income claim vs. payslip scan |
| Evidence Review   | Court ruling vs. submitted evidence photo |
| Licensing & Employee Verification | Credential claim vs. ID card photo |

---

## Architecture

User Input (claim text / document / image)
↓
Layer 1 — Document Ingestion (PDF, DOCX, TXT, OCR)
↓
Layer 2 — PII Protection (spaCy NER + OpenCV face/plate blur)
↓
Step A — Claim Extraction       → Groq LLaMA 3.3 70B (text)
Step B — Evidence Grounding     → Groq LLaMA 4 Scout 17B (vision)
Step C — Verdict Synthesis      → Deterministic Python rules
Step D — Explanation Generation → Groq LLaMA 3.3 70B (text)
↓
ComplianceReport (verdicts + confidence + explanations)
↓
PDF / JSON export

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/CodeWithBigboss-source/MultiModal-Lie-of-Omission-Detector-for-Compliance-Documents.git
cd MultiModal-Lie-of-Omission-Detector-for-Compliance-Documents
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API key

```bash
copy .env.example .env
```

Edit `.env` and add your Groq API key from https://console.groq.com

GROQ_API_KEY=your_key_here

### 3. Run Streamlit UI

```bash
streamlit run app.py
```

Opens at http://localhost:8501

### 4. Run FastAPI backend

```bash
uvicorn api:app --reload --port 8000
```

API docs at http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint       | Description                                 |
|--------|----------------|---------------------------------------------|
| GET    | /health        | System health check                         |
| GET    | /domains       | List available domains                      |
| POST   | /analyze       | Single document analysis                    |
| POST   | /analyze/multi | Multi-document with cross-doc contradiction |
| GET    | /report/{id}   | Retrieve stored report                      |
| GET    | /reports       | List all stored reports                     |

### Example API call

```bash
curl -X POST http://localhost:8000/analyze \
  -F "domain=vehicle_insurance" \
  -F "claim_text=My left front door is severely damaged" \
  -F "image=@data/sample_images/CL001.jpg"
```

---

## Run Evaluation

```bash
python scripts/run_eval.py
python scripts/run_eval.py --domain vehicle_insurance
python scripts/run_eval.py --max_cases 5
```

Place gold eval images in `data/sample_images/` matching filenames 
in `data/gold_eval/cases.json`.

---

## Project Structure

├── app.py                          # Streamlit UI
├── api.py                          # FastAPI backend
├── requirements.txt
├── .env.example
├── data/
│   ├── sample_docs/                # Sample claim documents
│   ├── sample_images/              # Evidence images
│   └── gold_eval/
│       └── cases.json              # Hand-labeled eval cases
├── src/
│   ├── utils/
│   │   ├── schemas.py              # Pydantic data contracts
│   │   └── model_client.py         # Groq API clients
│   ├── extraction/
│   │   └── claim_extraction.py     # Step A
│   ├── grounding/
│   │   └── image_grounding.py      # Step B
│   ├── verdict/
│   │   └── synthesize.py           # Step C
│   ├── explanation/
│   │   └── generate.py             # Step D
│   ├── pipeline.py                 # Orchestrator
│   ├── ingestion/
│   │   └── document_loader.py      # PDF/DOCX/TXT parser
│   ├── pii/
│   │   └── detector.py             # PII masking + image blur
│   ├── classification/
│   │   └── document_classifier.py  # Auto domain detection
│   ├── reasoning/
│   │   └── cross_document.py       # Cross-doc contradiction
│   ├── reporting/
│   │   └── report_generator.py     # PDF export
│   └── eval/
│       └── evaluator.py            # Accuracy measurement
└── scripts/
└── run_eval.py                 # CLI eval runner

---

## Technologies Used

| Layer               | Technology                        |
|---------------------|-----------------------------------|
| LLM (Steps A, D)    | Groq — LLaMA 3.3 70B Versatile    |
| Vision LLM (Step B) | Groq — LLaMA 4 Scout 17B          |
| PII Detection       | spaCy en_core_web_sm              |
| Image PII           | OpenCV Haar Cascades              |
| Document Parsing    | pypdf, python-docx                |
| OCR                 | Tesseract + pdf2image             |
| Schema Validation   | Pydantic v2                       |
| Retry Logic         | Tenacity                          |
| Report Generation   | fpdf2                             |
| Frontend            | Streamlit                         |
| Backend             | FastAPI + Uvicorn                 |
| Privacy Pattern     | Structure-preserving tokenization |

---

## Privacy Design

All PII is masked **before** any API call:

- Names → `[PERSON_1]`, `[PERSON_2]`
- IDs → `[ID_1]`
- Dates → `[DATE_1]`
- Faces in images → Gaussian blur
- License plates → Gaussian blur

Real values are stored in a local session registry and 
restored in the final report. No personal data ever 
reaches any external API.

---

## Datasets Used for Evaluation

| Domain            | Dataset                      |
|-------------------|------------------------------|
| Vehicle Insurance | CarDD (car damage detection) |
| Health Insurance  | NIH Chest X-ray, MedNLI      |
| Loan Application  | Home Credit Default Risk     |
| Evidence Review   | ECHR, AVerITeC               |
| Licensing         | MIDV-500 (ID documents)      |
#Mostly DATASET was half original and half synthetic

---

## Authors

**Malik Ahsan Nasar** — Generative AI Intern  
Genesys Research Lab, FAST NUCES, Islamabad  
GitHub: github.com/CodeWithBigboss-source  
LinkedIn: linkedin.com/in/malikahsannasar