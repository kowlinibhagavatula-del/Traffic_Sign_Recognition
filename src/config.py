from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data" / "GTSRB" / "Train"

# Model directory
MODEL_DIR = BASE_DIR / "models"

# Reports directory
REPORT_DIR = BASE_DIR / "reports"

# Model path
MODEL_PATH = MODEL_DIR / "traffic_sign_cnn.keras"

# Class names path
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

# Image settings
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3

# Training settings
BATCH_SIZE = 64
EPOCHS = 15
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Number of traffic-sign classes
NUM_CLASSES = 43

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)