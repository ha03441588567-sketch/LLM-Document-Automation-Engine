"""
Orchestrates: preprocessing -> model extraction -> postprocessing/mapping
into the ExtractedDocument schema.
"""
from PIL import Image

from app import config
from app.schemas import ExtractedDocument, LineItem
from app.inference import donut_extractor, layoutlm_extractor


def run_extraction(image: Image.Image) -> tuple[ExtractedDocument, str]:
    model_used = config.EXTRACTOR_MODEL

    if model_used == "layoutlm":
        result = layoutlm_extractor.extract(image)
        doc = _map_layoutlm_output(result)
    else:
        result = donut_extractor.extract(image)
        doc = _map_donut_output(result)

    return doc, model_used


def _map_donut_output(result: dict) -> ExtractedDocument:
    parsed = result.get("parsed", {}) or {}

    line_items = []
    for item in parsed.get("menu", []) or parsed.get("line_items", []) or []:
        line_items.append(
            LineItem(
                description=item.get("nm") or item.get("description"),
                quantity=item.get("cnt") or item.get("quantity"),
                unit_price=item.get("price") or item.get("unit_price"),
                total=item.get("total") or item.get("total_price"),
            )
        )

    return ExtractedDocument(
        vendor_name=parsed.get("vendor") or parsed.get("store_name"),
        invoice_number=parsed.get("invoice_no") or parsed.get("num"),
        invoice_date=parsed.get("date"),
        po_number=parsed.get("po_number"),
        total_amount=parsed.get("total", {}).get("total_price") if isinstance(parsed.get("total"), dict) else parsed.get("total"),
        currency=parsed.get("currency"),
        line_items=line_items,
        raw_model_output=result.get("raw_output"),
        confidence=None,
    )


def _map_layoutlm_output(result: dict) -> ExtractedDocument:
    return ExtractedDocument(
        raw_model_output=result.get("raw_output"),
        confidence=None,
    )
