import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'kagglecatsanddogs_3367a', 'PetImages')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'cat_dog_classifier.h5')

# Training parameters
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2

# Class names
CLASS_NAMES = ['cat', 'dog']