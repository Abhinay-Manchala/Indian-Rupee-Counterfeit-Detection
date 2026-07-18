"""
Dataset Loading Module

This module loads the preprocessed training and testing datasets,
resizes the images, normalizes pixel values, and returns NumPy arrays
for model training and evaluation.
"""

import os
import cv2
import numpy as np

from config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    IMAGE_SIZE,
    CLASS_LABELS
)

def load_dataset(folder_path):
    """
    Load images and labels from the specified dataset folder.

    Args:
        folder_path (str): Path to the dataset folder.

    Returns:
        tuple:
            images (numpy.ndarray): Array of images.
            labels (numpy.ndarray): Array of corresponding labels.
    """

    images = []
    labels = []

    for category, label in CLASS_LABELS.items():

        category_path = os.path.join(folder_path, category)

        if not os.path.exists(category_path):
            print(f"Warning: {category_path} not found.")
            continue

        for denomination in os.listdir(category_path):

            denomination_path = os.path.join(category_path, denomination)

            if not os.path.isdir(denomination_path):
                continue

            for image_name in os.listdir(denomination_path):

                image_path = os.path.join(denomination_path, image_name)

                image = cv2.imread(image_path)

                if image is None:
                    continue

                # Resize image
                image = cv2.resize(image, IMAGE_SIZE)

                # Normalize pixel values (0-255 → 0-1)
                image = image.astype("float32") / 255.0

                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)


if __name__ == "__main__":

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    X_train, y_train = load_dataset(TRAIN_DATASET_PATH)
    X_test, y_test = load_dataset(TEST_DATASET_PATH)

    print("\nDataset Loaded Successfully!")
    print("=" * 50)

    print(f"Training Images : {X_train.shape}")
    print(f"Training Labels : {y_train.shape}")

    print()

    print(f"Testing Images  : {X_test.shape}")
    print(f"Testing Labels  : {y_test.shape}")

    print()

    print("Pixel Value Range")
    print(f"Minimum : {X_train.min()}")
    print(f"Maximum : {X_train.max()}")

    print("=" * 50)