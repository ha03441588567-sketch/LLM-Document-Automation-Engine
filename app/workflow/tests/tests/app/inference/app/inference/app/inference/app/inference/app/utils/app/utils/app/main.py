from fastapi import FastAPI, UploadFile, File, HTTPException

from app import config
from app.schemas import ExtractionResponse
from app.utils.preprocessing import load_images_from_upload, resize_for_model
from app.inference.pipeline import run_extraction
from app.workflow.automation import trigger_workflow

app = FastAPI(
    title="LLM Document Automation Engine",
    description="Intelligent Document Processing for invoices & logistics documents",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model": config.EXTRACTOR_MODEL, "device": config.DEVICE}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(file: UploadFile = File(...)):
    contents = await file.read()

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {config.MAX_UPLOAD_MB}MB limit")

    try:
        images = load_images_from_upload(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not images:
        raise HTTPException(status_code=400, detail="Could not read any pages from the document")

    image = resize_for_model(images[0])

    try:
        extracted_doc, model_used = run_extraction(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    workflow_triggered = trigger_workflow(extracted_doc, file.filename)

    return ExtractionResponse(
        success=True,
        filename=file.filename,
        model_used=model_used,
        extracted=extracted_doc,
        workflow_triggered=workflow_triggered,
    )
