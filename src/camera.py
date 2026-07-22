"""
Live Webcam Counterfeit Detection

This module captures live webcam frames, crops the
Region of Interest (ROI), performs prediction using
the trained CNN model, and displays the result.
"""

import cv2
import numpy as np

from predict import (
    load_prediction_model,
    get_prediction_result,
)

from config import IMAGE_SIZE


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess ROI for CNN prediction.
    """

    frame = cv2.resize(frame, IMAGE_SIZE)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = frame.astype("float32") / 255.0

    frame = np.expand_dims(
        frame,
        axis=0
    )

    return frame


def note_present(roi: np.ndarray) -> bool:
    """
    Check whether a banknote is present inside the ROI.
    """

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    edge_pixels = cv2.countNonZero(edges)

    return edge_pixels > 2500


def start_webcam() -> None:

    print("=" * 50)
    print("Opening Webcam...")
    print("=" * 50)

    model = load_prediction_model()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Unable to open webcam.")
        return

    print("Webcam Started Successfully!")
    print("Place the entire note inside the green box.")
    print("Press 'Q' to Exit.\n")

    prediction_text = "Waiting..."
    confidence_text = ""

    frame_count = 0

    while True:

        success, frame = camera.read()

        if not success:
            break

        height, width, _ = frame.shape

        # Slightly larger ROI
        roi_width = 460
        roi_height = 220

        x1 = (width - roi_width) // 2
        y1 = (height - roi_height) // 2

        x2 = x1 + roi_width
        y2 = y1 + roi_height

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Place Entire Note Here",
            (x1 + 15, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        roi = frame[y1:y2, x1:x2]

        frame_count += 1

        # Predict every 15th frame
        if frame_count % 15 == 0:

            if note_present(roi):

                # Improve brightness & contrast
                enhanced_roi = cv2.convertScaleAbs(
                    roi,
                    alpha=1.15,
                    beta=8
                )

                processed_frame = preprocess_frame(enhanced_roi)

                prediction = model.predict(
                    processed_frame,
                    verbose=0
                )

                confidence = float(prediction[0][0])

                predicted_class, confidence_score = get_prediction_result(
                    confidence
                )

                prediction_text = f"Prediction : {predicted_class}"
                confidence_text = ""

            else:

                prediction_text = "No Note Detected"
                confidence_text = ""

        # Text color
        if "Real" in prediction_text:
            color = (0, 255, 0)

        elif "Fake" in prediction_text:
            color = (0, 0, 255)

        else:
            color = (0, 255, 255)

        cv2.putText(
            frame,
            prediction_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )


        cv2.imshow(
            "Indian Rupee Counterfeit Detection System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()


def main() -> None:
    start_webcam()


if __name__ == "__main__":
    main()