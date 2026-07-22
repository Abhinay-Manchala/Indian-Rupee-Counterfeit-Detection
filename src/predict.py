"""
Prediction Module

Loads the trained CNN model and performs
prediction on an input currency note image.
"""

import numpy as np

from tensorflow.keras.models import (
    load_model,
    Sequential,
)

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

    Returns
    -------
    Sequential
        Loaded TensorFlow model.
    """

    try:

        model = load_model(
            MODEL_SAVE_PATH
        )

        print("\nModel Loaded Successfully!\n")

        return model

    except Exception as error:

        print(f"\nError loading model: {error}")

        raise


def preprocess_image(
    image_path: str
) -> np.ndarray:
    """
    Load and preprocess an image.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    np.ndarray
    """

    try:

        img = image.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

    except Exception as error:

        print(f"\nError loading image: {error}")

        raise

    img_array = image.img_to_array(
        img
    )

    img_array = img_array.astype(
        "float32"
    )

    img_array /= 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

def get_prediction_result(
    confidence: float,
) -> tuple[str, float]:
    """
    Determine prediction result.

    Parameters
    ----------
    confidence : float

    Returns
    -------
    tuple[str, float]
    """

    if confidence >= CONFIDENCE_THRESHOLD:

        predicted_class = CLASS_NAMES[1]
        confidence_score = confidence * 100

    else:

        predicted_class = CLASS_NAMES[0]
        confidence_score = (1 - confidence) * 100

    confidence_score = round(
        confidence_score,
        2
    )

    return (
        predicted_class,
        confidence_score
    )


def predict_note(
    model: Sequential,
    processed_image: np.ndarray,
) -> tuple[str, float]:
    """
    Predict whether the note is Real or Fake.

    Parameters
    ----------
    model : Sequential

    processed_image : np.ndarray

    Returns
    -------
    tuple[str, float]
    """

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    confidence = float(
        prediction[0][0]
    )

    return get_prediction_result(
        confidence
    )


def display_prediction(
    predicted_class: str,
    confidence_score: float,
) -> None:
    """
    Display only the prediction result.
    """

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"Prediction : {predicted_class}")
    print("=" * 50)


def main() -> None:
    """
    Execute prediction workflow.
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