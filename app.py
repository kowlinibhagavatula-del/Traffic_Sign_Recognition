import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    return load_model("models/traffic_sign_cnn.keras")


model = load_trained_model()


# --------------------------------------------------
# Traffic Sign Classes
# --------------------------------------------------

class_names = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to the same size used during training
    image = image.resize((32, 32))

    # Convert to NumPy array
    image = np.array(image)

    # Normalize pixel values
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🚦 Traffic Sign Recognition")

st.write(
    "Upload a traffic sign image and the CNN model "
    "will predict the traffic sign."
)

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=300
    )

    if st.button("🔍 Predict Traffic Sign"):

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image, verbose=0)

        predicted_class = np.argmax(predictions[0])

        confidence = float(
            predictions[0][predicted_class] * 100
        )

        predicted_sign = class_names[predicted_class]

        st.success(
            f"Prediction: {predicted_sign}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.write(
            f"**Class ID:** {predicted_class}"
        )

        # ------------------------------------------
        # Top 5 Predictions
        # ------------------------------------------

        st.subheader("Top 5 Predictions")

        top_5_indices = np.argsort(
            predictions[0]
        )[-5:][::-1]

        for index in top_5_indices:

            probability = (
                predictions[0][index] * 100
            )

            st.write(
                f"**{class_names[index]}** — "
                f"{probability:.2f}%"
            )