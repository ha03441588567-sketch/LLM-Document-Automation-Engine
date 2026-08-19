from typing import Optional
from pydantic import BaseModel


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    total: Optional[str] = None


class ExtractedDocument(BaseModel):
    document_type: str = "invoice"
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    po_number: Optional[str] = None
    total_amount: Optional[str] = None
    currency: Optional[str] = None
    line_items: list[LineItem] = []
    raw_model_output: Optional[str] = None
    confidence: Optional[float] = None


class ExtractionResponse(BaseModel):
    success: bool
    filename: str
    model_used: str
    extracted: ExtractedDocument
    workflow_triggered: bool
