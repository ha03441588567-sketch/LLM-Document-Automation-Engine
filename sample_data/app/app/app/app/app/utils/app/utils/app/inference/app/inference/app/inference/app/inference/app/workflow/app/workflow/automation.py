"""
Downstream automation: pushes extracted document JSON to a configured
webhook (Zapier / Make.com / n8n / ERP API) so the whole pipeline is
hands-off after extraction.
"""
import requests

from app import config
from app.schemas import ExtractedDocument


def trigger_workflow(document: ExtractedDocument, filename: str) -> bool:
    if not config.WORKFLOW_WEBHOOK_URL:
        return False

    payload = {
        "filename": filename,
        "document": document.model_dump(),
    }

    try:
        response = requests.post(config.WORKFLOW_WEBHOOK_URL, json=payload, timeout=10)
        return response.ok
    except requests.RequestException:
        return False
