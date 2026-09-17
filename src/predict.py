import sys

import cv2
import numpy as np

from tensorflow.keras.models import load_model

from src.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMG_HEIGHT,
    IMG_WIDTH
)


def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            line.strip()
            for line in file.readlines()
        ]


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    # Normalize
    image = image.astype(
        np.float32
    ) / 255.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


def predict(image_path):

    print("Loading model...")

    model = load_model(MODEL_PATH)

    class_names = load_class_names()

    image = preprocess_image(
        image_path
    )

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    predicted_class = np.argmax(
        probabilities
    )

    confidence = probabilities[
        predicted_class
    ]

    print("\nPrediction Result")
    print("-----------------")

    print(
        "Class ID:",
        predicted_class
    )

    print(
        "Traffic Sign:",
        class_names[predicted_class]
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    return (
        predicted_class,
        class_names[predicted_class],
        confidence
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python -m src.predict "
            "path_to_image"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict(image_path)