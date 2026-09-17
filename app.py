import cv2
import numpy as np
import streamlit as st

from PIL import Image
from tensorflow.keras.models import load_model

from src.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMG_HEIGHT,
    IMG_WIDTH
)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)


# -----------------------------
# Load Class Names
# -----------------------------

@st.cache_resource
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


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_cnn_model():

    return load_model(
        MODEL_PATH
    )


# -----------------------------
# Preprocess Image
# -----------------------------

def preprocess_image(image):

    image = np.array(image)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    image = cv2.resize(
        image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    # Convert back to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
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


# -----------------------------
# Application
# -----------------------------

st.title("🚦 Traffic Sign Recognition")

st.write(
    "Upload a traffic sign image and "
    "the CNN model will predict the sign."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=300
    )

    st.divider()

    if st.button(
        "🔍 Predict Traffic Sign"
    ):

        try:

            model = load_cnn_model()
            class_names = load_class_names()

            processed_image = preprocess_image(
                image
            )

            probabilities = model.predict(
                processed_image,
                verbose=0
            )[0]

            predicted_class = np.argmax(
                probabilities
            )

            confidence = probabilities[
                predicted_class
            ]

            predicted_name = class_names[
                predicted_class
            ]

            st.success(
                f"Prediction: {predicted_name}"
            )

            st.info(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

            # Show top 5 predictions
            st.subheader(
                "Top 5 Predictions"
            )

            top_indices = np.argsort(
                probabilities
            )[-5:][::-1]

            for index in top_indices:

                st.write(
                    f"**{class_names[index]}** "
                    f"- "
                    f"{probabilities[index] * 100:.2f}%"
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )