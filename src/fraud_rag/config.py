from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
DEFAULT_HF_CLASSIFIER = "distilbert-base-uncased"
