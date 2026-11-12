"""Shared paths and settings for the eGeMAPS -> logistic regression pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_ID = "ylacombe/expresso"
CONFIG = "read"
SPLIT = "train"
CACHE = ROOT / ".hf_cache"

# The 7 project styles. All other read-set styles (emphasis, essentials,
# longform, singing) are dropped; team decision 2026-11-09.
STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad", "whisper"]

CLIPS_CSV = ROOT / "clips.csv"
FEATURES = ROOT / "features" / "egemaps.parquet"
FOLDS_CSV = ROOT / "splits" / "folds.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

TARGET_SR = 16_000
SEED = 42
# Logistic regression uses one fixed C for every fold, so coefficients and SHAP
# values are comparable across folds (decision 2026-11-11). C_GRID is only scanned
# as a sensitivity check (results/c_sensitivity_macro_f1.json); it does not pick C.
C_FIXED = 1.0
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
