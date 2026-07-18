import os
import cv2
import numpy as np

TRAIN_PATH = "dataset/processed/train"
TEST_PATH = "dataset/processed/test"

IMG_SIZE = 224

classes = {
    "fake": 0,
    "real": 1
}


def load_dataset(folder_path):

    images = []
    labels = []

    for category, label in classes.items():

        category_path = os.path.join(folder_path, category)

        if not os.path.exists(category_path):
            continue

        for denomination in os.listdir(category_path):

            denomination_path = os.path.join(category_path, denomination)

            for image_name in os.listdir(denomination_path):

                image_path = os.path.join(denomination_path, image_name)

                image = cv2.imread(image_path)

                if image is None:
                    continue

                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                # Normalize pixel values
                image = image.astype("float32") / 255.0

                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)


# Load datasets
X_train, y_train = load_dataset(TRAIN_PATH)
X_test, y_test = load_dataset(TEST_PATH)

print("=" * 50)
print("Dataset Loaded Successfully!")
print("=" * 50)

print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print()

print("Testing Images  :", X_test.shape)
print("Testing Labels  :", y_test.shape)

print()
print("Pixel Value Range")
print("Minimum :", X_train.min())
print("Maximum :", X_train.max())