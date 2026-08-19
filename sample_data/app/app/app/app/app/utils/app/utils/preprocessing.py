"""
Document preprocessing: converts uploaded PDFs/images into a list of
PIL Image objects ready for model inference.
"""
import io
from PIL import Image
from pdf2image import convert_from_bytes


def load_images_from_upload(file_bytes: bytes, filename: str) -> list[Image.Image]:
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        pages = convert_from_bytes(file_bytes, dpi=200)
        return [page.convert("RGB") for page in pages]

    if ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return [image]

    raise ValueError(f"Unsupported file type: .{ext}")


def resize_for_model(image: Image.Image, max_dim: int = 1600) -> Image.Image:
    """Downscale large scans to keep inference fast, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))
