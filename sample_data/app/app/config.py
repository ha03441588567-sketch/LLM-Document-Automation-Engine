"""Central settings, loaded from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()

EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "donut")  # "donut" or "layoutlm"

DONUT_MODEL_ID = os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2")
LAYOUTLM_MODEL_ID = os.getenv("LAYOUTLM_MODEL_ID", "microsoft/layoutlmv3-base")

DEVICE = os.getenv("DEVICE", "cpu")

WORKFLOW_WEBHOOK_URL = os.getenv("WORKFLOW_WEBHOOK_URL", "")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
