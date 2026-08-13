import yaml
from pathlib import Path

# src/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"config.yaml not found at {CONFIG_PATH}. "
        f"Copy config.example.yaml to config.yaml and fill in your paths."
    )

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

# --- Base directories ---
DATA_DIR = (PROJECT_ROOT / _cfg["paths"]["data_dir"]).resolve()
MODELS_DIR = (PROJECT_ROOT / _cfg["paths"]["models_dir"]).resolve()

# --- Data files / subfolders ---
BRSET_LABELS_CSV = DATA_DIR / _cfg["data"]["brset_labels_csv"]
CAPTIONS_CSV = DATA_DIR / _cfg["data"]["captions_csv"]
CLASSIFICATION_CAPTIONS_CSV = DATA_DIR / _cfg["data"]["classification_captions_csv"]
IMAGES_DIR = DATA_DIR / _cfg["data"]["images_dir"]
TEST_IMAGES_DIR = DATA_DIR / _cfg["data"]["test_images_dir"]
DISTRIBUTIONS_DIR = DATA_DIR / _cfg["data"]["distributions_dir"]

# --- Models ---
OD_MODEL_PATH = MODELS_DIR / _cfg["models"]["od_model"]
FOVEA_MODEL_PATH = MODELS_DIR / _cfg["models"]["fovea_model"]

# --- Raw dataset (external source, e.g. BRSET download) ---
RAW_BRSET_DIR = Path(_cfg["raw_data"]["dataset_dir"])
RAW_BRSET_LABELS_CSV = RAW_BRSET_DIR / _cfg["raw_data"]["labels_file"]
RAW_BRSET_IMAGES_DIR = RAW_BRSET_DIR / _cfg["raw_data"]["images_subdir"]