"""
Model Training Module

This script:
1. Loads the dataset
2. Builds the CNN model
3. Trains the model
4. Saves the best model
5. Saves training history
"""

import json

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from dataset_loader import load_dataset
from model import build_model
from config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    EPOCHS,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
)


def train_model():
    """Train the CNN model."""

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    # Load datasets
    X_train, y_train = load_dataset(TRAIN_DATASET_PATH)
    X_test, y_test = load_dataset(TEST_DATASET_PATH)

    print("Dataset Loaded Successfully!\n")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
    print("=" * 50)

    # Build CNN model
    model = build_model()

    # Save the best model
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # Stop training if validation accuracy stops improving
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print("=" * 50)
    print("Training Started...")
    print("=" * 50)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping],
        verbose=1,
    )

    # Save training history
    with open("results/training_history.json", "w") as file:
        json.dump(history.history, file, indent=4)

    print("\nTraining history saved successfully!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    return history


if __name__ == "__main__":
    train_model()