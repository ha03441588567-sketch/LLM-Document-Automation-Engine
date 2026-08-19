# 📄 LLM Document Automation Engine (IDP)

Intelligent Document Processing system that extracts structured data from **invoices and logistics documents** (PDFs/scanned images) and feeds it directly into automated downstream workflows — no manual data entry.

## What it does

1. Accepts an uploaded invoice/logistics document (PDF or image).
2. Runs it through a layout-aware vision-language model (Donut or LayoutLM) to extract structured fields: invoice number, vendor, line items, totals, dates, PO numbers.
3. Validates and normalizes the extracted data.
4. Pushes the structured JSON to a downstream automation workflow (webhook, ERP, database).

## Tech Stack
- Extraction models: Hugging Face — Donut, LayoutLMv3
- ML runtime: PyTorch
- API layer: FastAPI
- Document handling: pdf2image, Pillow

## Project Structure
idp-engine/
├── app/
│ ├── main.py
│ ├── config.py
│ ├── schemas.py
│ ├── inference/
│ │ ├── donut_extractor.py
│ │ ├── layoutlm_extractor.py
│ │ └── pipeline.py
│ ├── workflow/
│ │ └── automation.py
│ └── utils/
│ └── preprocessing.py
├── tests/test_api.py
├── sample_data/README.md
├── requirements.txt
└── .env.example


## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` and upload a document via `POST /extract`.

## Choosing a model
- **Donut** — OCR-free, fastest to set up, good default for invoices/receipts.
- **LayoutLMv3** — OCR + layout understanding, more accurate on complex documents, needs Tesseract installed.

Switch via `EXTRACTOR_MODEL=donut` or `layoutlm` in `.env`.

## Deploy live (free)
1. Push to GitHub
2. Deploy on Render or Railway (free tier): build `pip install -r requirements.txt`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Automation hook
`app/workflow/automation.py` posts extracted JSON to `WORKFLOW_WEBHOOK_URL` — point at Zapier/Make.com/n8n/your ERP.

## License
MIT
