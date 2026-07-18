"""
Image Preprocessing Module

This module preprocesses the Indian Rupee currency note dataset by:
- Reading images from the raw dataset
- Resizing them to a fixed size
- Saving the processed images into a new directory
"""

import os
import cv2

from config import (
    RAW_DATASET_PATH,
    TRAIN_DATASET_PATH,
    IMAGE_SIZE
)

def preprocess_images():
    """
    Preprocess all images in the dataset.

    The function reads images from the raw dataset,
    resizes them to the specified image size,
    and saves them into the processed dataset folder.
    """

    print("=" * 60)
    print("Indian Rupee Counterfeit Detection - Image Preprocessing")
    print("=" * 60)

    total_processed = 0
    total_skipped = 0

    # Loop through fake and real folders
    for category in os.listdir(RAW_DATASET_PATH):

        category_path = os.path.join(RAW_DATASET_PATH, category)

        if not os.path.isdir(category_path):
            continue

        # Loop through denomination folders
        for denomination in os.listdir(category_path):

            input_path = os.path.join(category_path, denomination)

            if not os.path.isdir(input_path):
                continue

            output_path = os.path.join(
                TRAIN_DATASET_PATH,
                category,
                denomination
            )

            os.makedirs(output_path, exist_ok=True)

            images = os.listdir(input_path)

            print(f"\nProcessing: {category}/{denomination}")
            print(f"Images Found : {len(images)}")

            processed = 0
            skipped = 0

            for image_name in images:

                image_path = os.path.join(input_path, image_name)

                image = cv2.imread(image_path)

                if image is None:
                    skipped += 1
                    total_skipped += 1
                    continue

                # Resize image
                image = cv2.resize(image, IMAGE_SIZE)

                # Save processed image
                save_path = os.path.join(output_path, image_name)
                cv2.imwrite(save_path, image)

                processed += 1
                total_processed += 1

            print(f"Processed    : {processed}")
            print(f"Skipped      : {skipped}")

    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Total Images Processed : {total_processed}")
    print(f"Total Images Skipped   : {total_skipped}")
    print("Preprocessing Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_images()