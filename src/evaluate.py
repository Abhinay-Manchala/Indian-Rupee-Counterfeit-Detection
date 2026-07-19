"""
Model Evaluation Module

This script:
1. Loads the trained model
2. Loads the test dataset
3. Predicts counterfeit notes
4. Evaluates model performance
5. Saves evaluation results
"""

import csv

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset_loader import load_dataset
from config import (
    TEST_DATASET_PATH,
    MODEL_SAVE_PATH,
)


def evaluate_model():
    """Evaluate the trained CNN model."""

    print("=" * 50)
    print("Loading Model...")
    print("=" * 50)

    # Load trained model
    model = load_model(MODEL_SAVE_PATH)

    print("Model Loaded Successfully!\n")

    print("=" * 50)
    print("Loading Test Dataset...")
    print("=" * 50)

    # Load test dataset
    X_test, y_test = load_dataset(TEST_DATASET_PATH)

    print(f"Testing Samples : {len(X_test)}")
    print("=" * 50)

    print("Predicting...")

    # Predict labels
    predictions = model.predict(X_test)

    print("Prediction Completed!\n")

    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)

    print("=" * 50)
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print("=" * 50)

    # Save test accuracy
    with open("results/test_accuracy.txt", "w") as file:
        file.write(f"Test Accuracy : {accuracy * 100:.2f}%\n")

    # Generate classification report
    print("\nClassification Report")

    report = classification_report(
        y_test,
        predicted_labels,
        target_names=["Fake", "Real"]
    )

    print(report)

    # Save classification report
    with open("results/classification_report.txt", "w") as file:
        file.write(report)

    # Generate confusion matrix
    print("Confusion Matrix")

    matrix = confusion_matrix(
        y_test,
        predicted_labels
    )

    print(matrix)

    # Save confusion matrix
    with open("results/confusion_matrix.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(matrix)

    print("\nEvaluation Results Saved Successfully!")
    print("Location : results/")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()