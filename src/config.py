"""
Project Configuration

This module stores common constants used throughout the project.
Keeping all configurable values in one place makes the project
easier to maintain.
"""

# Image settings
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Dataset paths
RAW_DATASET_PATH = "dataset/train"
TRAIN_DATASET_PATH = "dataset/processed/train"
TEST_DATASET_PATH = "dataset/processed/test"

# Dataset split
TEST_RATIO = 0.20
RANDOM_SEED = 42

# Class labels
CLASS_LABELS = {
    "fake": 0,
    "real": 1
}

# Training configuration
EPOCHS = 20
BATCH_SIZE = 32

# Model save path
MODEL_SAVE_PATH = "models/best_model.keras"