from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCALER_DIR = DATA_DIR / "scalers"

SWEEP_CONFIG_FILE = ROOT_DIR / "sweep_config.yml"
RAW_LIGHTCURVES_FILE = RAW_DATA_DIR / "training_set.csv"
RAW_METADATA_FILE = RAW_DATA_DIR / "training_set_metadata.csv"
PROCESSED_TRAINING_SET_FILE = PROCESSED_DATA_DIR / "training_set.pkl"

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

N_BANDS = 6
METADATA_INPUT_SIZE = 20
LIGHTCURVE_INPUT_SIZE = 6
NUM_CLASSES = 14
