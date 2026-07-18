"""
CNN Model Module

This module defines the Convolutional Neural Network (CNN) architecture
used to classify Indian currency notes as Real or Fake.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)

from config import INPUT_SHAPE

def build_model():
    """
    Build and compile the CNN model.

    Returns:
        tensorflow.keras.Model: Compiled CNN model.
    """

    model = Sequential([

        # Input layer
        Input(shape=INPUT_SHAPE),

        # First Convolution Block
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu"
        ),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolution Block
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),
        MaxPooling2D(pool_size=(2, 2)),

        # Feature Extraction
        GlobalAveragePooling2D(),

        # Fully Connected Layer
        Dense(
            units=128,
            activation="relu"
        ),

        # Dropout Layer
        Dropout(0.5),

        # Output Layer
        Dense(
            units=1,
            activation="sigmoid"
        )
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    cnn_model = build_model()

    print("=" * 50)
    print("CNN Model Summary")
    print("=" * 50)

    cnn_model.summary()

    print("=" * 50)