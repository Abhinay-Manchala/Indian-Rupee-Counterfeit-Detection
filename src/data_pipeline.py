"""
Dataset Split Module

This module splits the processed dataset into training and testing sets.
A percentage of images from each category is moved from the training
folder to the testing folder.
"""

import os
import random
import shutil

from config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    TEST_RATIO,
    RANDOM_SEED
)


def split_dataset():
    """
    Split the processed dataset into training and testing sets.

    A fixed percentage of images is moved from the training folder
    to the testing folder while preserving the folder structure.
    """

    print("=" * 60)
    print("Dataset Split (80% Train | 20% Test)")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    total_train = 0
    total_test = 0

    for category in os.listdir(TRAIN_DATASET_PATH):

        category_path = os.path.join(TRAIN_DATASET_PATH, category)

        if not os.path.isdir(category_path):
            continue

        for denomination in os.listdir(category_path):

            train_folder = os.path.join(category_path, denomination)

            if not os.path.isdir(train_folder):
                continue

            test_folder = os.path.join(
                TEST_DATASET_PATH,
                category,
                denomination
            )

            os.makedirs(test_folder, exist_ok=True)

            images = os.listdir(train_folder)

            random.shuffle(images)

            test_size = int(len(images) * TEST_RATIO)

            test_images = images[:test_size]

            for image_name in test_images:

                source_path = os.path.join(train_folder, image_name)
                destination_path = os.path.join(test_folder, image_name)

                shutil.move(source_path, destination_path)

            train_count = len(images) - test_size

            total_train += train_count
            total_test += test_size

            print(f"\nCategory          : {category}/{denomination}")
            print(f"Total Images      : {len(images)}")
            print(f"Training Images   : {train_count}")
            print(f"Testing Images    : {test_size}")

    print("\n" + "=" * 60)
    print("Dataset Split Summary")
    print("=" * 60)
    print(f"Total Training Images : {total_train}")
    print(f"Total Testing Images  : {total_test}")
    print("Dataset Split Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset()