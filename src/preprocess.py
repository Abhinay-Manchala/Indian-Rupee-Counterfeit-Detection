import os
import cv2

# Input and Output folders
INPUT_FOLDER = "dataset/train"
OUTPUT_FOLDER = "dataset/processed/train"

# Resize dimensions
IMAGE_SIZE = (224, 224)

print("=" * 60)
print("Indian Rupee Counterfeit Detection - Image Preprocessing")
print("=" * 60)

total_processed = 0

# Loop through real and fake
for category in os.listdir(INPUT_FOLDER):

    category_path = os.path.join(INPUT_FOLDER, category)

    if not os.path.isdir(category_path):
        continue

    # Loop through 100 and 500 folders
    for denomination in os.listdir(category_path):

        input_path = os.path.join(category_path, denomination)
        output_path = os.path.join(OUTPUT_FOLDER, category, denomination)

        # Create output folder if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        images = os.listdir(input_path)

        print(f"\nProcessing {category}/{denomination}")
        print(f"Images Found: {len(images)}")

        processed = 0

        for image_name in images:

            image_file = os.path.join(input_path, image_name)

            image = cv2.imread(image_file)

            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, IMAGE_SIZE)

            # Save image
            save_path = os.path.join(output_path, image_name)
            cv2.imwrite(save_path, image)

            processed += 1
            total_processed += 1

        print(f"Processed: {processed}")

print("\n" + "=" * 60)
print(f"Total Images Processed: {total_processed}")
print("Preprocessing Completed Successfully!")
print("=" * 60)