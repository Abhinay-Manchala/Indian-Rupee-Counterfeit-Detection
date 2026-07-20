"""
Prediction Module

This module performs inference using the trained CNN model.

Workflow:
1. Load the trained CNN model.
2. Load the input currency note image.
3. Preprocess the image for prediction.
4. Predict whether the note is Real or Fake.
5. Display the prediction result with confidence score.
"""

import numpy as np

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image

from config import (
    IMAGE_SIZE,
    MODEL_SAVE_PATH,
    PREDICTION_IMAGE_PATH,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
)


def load_prediction_model() -> Sequential:
    """
    Load the trained CNN model.

    Returns:
        Sequential: Loaded TensorFlow CNN model.
    """

    print("=" * 50)
    print("Loading Model...")
    print("=" * 50)

    try:
        model = load_model(MODEL_SAVE_PATH)

        print("Model Loaded Successfully!\n")

        return model

    except FileNotFoundError:
        print("Error: Model file not found.")
        raise

    except Exception as error:
        print(f"Error loading model: {error}")
        raise


def preprocess_image(
    image_path: str
) -> np.ndarray:
    """
    Load and preprocess an image for prediction.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """

    print("=" * 50)
    print("Loading Image...")
    print("=" * 50)

    try:
        img = image.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        print("Image Loaded Successfully!\n")

    except FileNotFoundError:
        print("Error: Image file not found.")
        raise

    except Exception as error:
        print(f"Error loading image: {error}")
        raise

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


def get_prediction_result(
    confidence: float,
) -> tuple[str, float]:
    """
    Determine the prediction result from the confidence score.

    Args:
        confidence (float): Raw confidence score produced by the model.

    Returns:
        tuple[str, float]:
            - Predicted class (Real or Fake).
            - Confidence score as a percentage.
    """

    if confidence >= CONFIDENCE_THRESHOLD:
        predicted_class = CLASS_NAMES[1]
        confidence_score = confidence * 100
    else:
        predicted_class = CLASS_NAMES[0]
        confidence_score = (1 - confidence) * 100

    return predicted_class, confidence_score


def predict_note(
    model: Sequential,
    processed_image: np.ndarray,
) -> tuple[str, float]:
    """
    Predict whether the currency note is Real or Fake.

    Args:
        model (Sequential): Loaded CNN model.
        processed_image (np.ndarray): Preprocessed input image.

    Returns:
        tuple[str, float]:
            - Predicted class (Real or Fake).
            - Confidence score as a percentage.
    """

    print("=" * 50)
    print("Predicting...")
    print("=" * 50)

    prediction = model.predict(processed_image)

    confidence = float(prediction[0][0])

    return get_prediction_result(confidence)


def display_prediction(
    predicted_class: str,
    confidence_score: float,
) -> None:
    """
    Display the prediction result.

    Args:
        predicted_class (str): Predicted class label.
        confidence_score (float): Prediction confidence percentage.

    Returns:
        None
    """

    print("\nPrediction Completed!\n")

    print("=" * 50)
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence_score:.2f}%")
    print("=" * 50)


def main() -> None:
    """
    Execute the complete prediction workflow.

    Returns:
        None
    """

    model = load_prediction_model()

    processed_image = preprocess_image(
        PREDICTION_IMAGE_PATH
    )

    predicted_class, confidence_score = predict_note(
        model,
        processed_image
    )

    display_prediction(
        predicted_class,
        confidence_score
    )


if __name__ == "__main__":
    main()