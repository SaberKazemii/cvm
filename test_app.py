import os
import requests
import torch
import numpy as np
import cv2
from PIL import Image
import streamlit as st

from cs_classifier import ResNet18Classifier  # Ensure this imports correctly
from transformations import preprocess_image  # Ensure this imports correctly

# Dropbox direct download link (ensure 'dl=1' for direct download)
MODEL_URL = "https://www.dropbox.com/scl/fi/cvz8yxl1872m8628mlxn1/best_model.pth?rlkey=mtw4t5gae21lxegh3k842nd66&dl=1"

@st.cache_data
def download_model(file_path="best_model.pth"):
    """
    Download the model file with a progress bar shown in the Streamlit app.
    Args:
        file_path: Path to save the model file.
    """
    if not os.path.exists(file_path):
        st.write("Downloading model... This may take a while.")
        try:
            response = requests.get(MODEL_URL, stream=True, timeout=60)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            total_size = int(response.headers.get('content-length', 0))
            bytes_downloaded = 0

            # Streamlit progress bar
            progress_bar = st.progress(0)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        progress = bytes_downloaded / total_size
                        progress_bar.progress(min(progress, 1.0))  # Update progress bar

            st.success("Model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading the model: {e}")
            raise RuntimeError("Failed to download the model file.")
    else:
        st.info("Model file already exists locally.")
    return file_path

@st.cache_resource
def initialize_model(file_path="best_model.pth"):
    """
    Initialize the model and load the weights.
    Args:
        file_path: Path to the model weights file.
    Returns:
        The initialized and loaded model.
    """
    try:
        st.write("Initializing the model...")
        model = ResNet18Classifier(num_classes=6, input_channels=3)
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        st.success("The trained deep learning model has been initialized successfully.")
        return model
    except Exception as e:
        st.error(f"Error during model initialization: {e}")
        raise SystemExit("Model initialization failed. Exiting application.")

# Download and initialize model
model_file_path = download_model()
model = initialize_model(model_file_path)

# Streamlit app UI
st.title("Age Stage Prediction App!")
st.write(
    "The AI model estimates the age of cervical vertebrae by analyzing radiographic images "
    "and mimicking the diagnostic accuracy of orthodontic specialists."
)
st.write("Upload an image, and the model will predict!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.write("Image uploaded successfully.")
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Handle grayscale images
        if len(image_np.shape) == 2:  # If grayscale, convert to 3 channels
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # Convert RGB to BGR for OpenCV compatibility
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Preprocess the image using the custom pipeline
        input_tensor = preprocess_image(image_cv).unsqueeze(0)  # Add batch dimension

        # Prediction
        st.write("Starting model prediction...")
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = prediction.argmax(dim=1).item()
        st.write(f"Predicted Class: {predicted_class + 1}")
    except Exception as e:
        st.error(f"Error during preprocessing or prediction: {e}")
