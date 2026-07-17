import os
import shutil
import random

# Source folder
SOURCE_FOLDER = "dataset/processed/train"

# Destination folder
DESTINATION_FOLDER = "dataset/processed/test"

# Test ratio
TEST_RATIO = 0.20

print("=" * 60)
print("Dataset Split (80% Train | 20% Test)")
print("=" * 60)

random.seed(42)

for category in os.listdir(SOURCE_FOLDER):

    category_path = os.path.join(SOURCE_FOLDER, category)

    if not os.path.isdir(category_path):
        continue

    for denomination in os.listdir(category_path):

        train_folder = os.path.join(category_path, denomination)

        test_folder = os.path.join(
            DESTINATION_FOLDER,
            category,
            denomination
        )

        os.makedirs(test_folder, exist_ok=True)

        images = os.listdir(train_folder)

        random.shuffle(images)

        test_size = int(len(images) * TEST_RATIO)

        test_images = images[:test_size]

        for image in test_images:

            src = os.path.join(train_folder, image)
            dst = os.path.join(test_folder, image)

            shutil.move(src, dst)

        print(f"\n{category}/{denomination}")
        print(f"Total Images     : {len(images)}")
        print(f"Moved to Test    : {test_size}")
        print(f"Remaining Train  : {len(images) - test_size}")

print("\n" + "=" * 60)
print("Dataset Split Completed Successfully!")
print("=" * 60)