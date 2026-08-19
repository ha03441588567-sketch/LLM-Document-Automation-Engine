"""
Donut (Document Understanding Transformer) extractor.
OCR-free: the model reads the image directly and generates structured
output as a token sequence, which we parse into JSON.
"""
import re
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from app import config

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = DonutProcessor.from_pretrained(config.DONUT_MODEL_ID)
        _model = VisionEncoderDecoderModel.from_pretrained(config.DONUT_MODEL_ID)
        _model.to(config.DEVICE)
        _model.eval()
    return _processor, _model


def extract(image: Image.Image) -> dict:
    processor, model = _load_model()

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(config.DEVICE)

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(config.DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            return_dict_in_generate=True,
        )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    parsed = processor.token2json(sequence)
    return {"raw_output": sequence, "parsed": parsed}
