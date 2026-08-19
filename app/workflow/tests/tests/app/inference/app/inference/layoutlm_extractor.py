"""
LayoutLMv3 extractor.
Combines OCR (via pytesseract, required system dependency: tesseract-ocr)
with layout + text understanding for token classification.

Note: LayoutLMv3-base is not fine-tuned for invoice field extraction out
of the box — for production use, fine-tune it on a labeled invoice dataset
and update LAYOUTLM_MODEL_ID accordingly.
"""
import pytesseract
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from app import config

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = LayoutLMv3Processor.from_pretrained(config.LAYOUTLM_MODEL_ID, apply_ocr=True)
        _model = LayoutLMv3ForTokenClassification.from_pretrained(config.LAYOUTLM_MODEL_ID)
        _model.to(config.DEVICE)
        _model.eval()
    return _processor, _model


def extract(image: Image.Image) -> dict:
    processor, model = _load_model()

    encoding = processor(image, return_tensors="pt", truncation=True)
    encoding = {k: v.to(config.DEVICE) for k, v in encoding.items()}

    import torch

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    labeled_tokens = [
        {"token": tok, "label": id2label.get(pred, str(pred))}
        for tok, pred in zip(tokens, predictions)
        if tok not in ("<s>", "</s>", "<pad>")
    ]

    raw_text = pytesseract.image_to_string(image)

    return {"raw_output": raw_text, "labeled_tokens": labeled_tokens}
