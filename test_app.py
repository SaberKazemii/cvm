import streamlit as st
from PIL import Image
import torch
from transformations import preprocess_image  # Import the preprocessing pipeline
from cs_classifier import ResNet18Classifier  # Import your model definition
import cv2
import numpy as np
# Streamlit app
st.title("Age Stage Prediction App!")
st.write("The AI model estimates the age of cervical vertebrae by analyzing radiographic images and mimicking the diagnostic accuracy of orthodontic specialists.")
# Initialize the model
# st.write("Initializing the model...")
import requests
import os


# Dropbox direct download link (ensure 'dl=1' for direct download)
MODEL_URL = "https://www.dropbox.com/scl/fi/cvz8yxl1872m8628mlxn1/best_model.pth?rlkey=mtw4t5gae21lxegh3k842nd66&dl=1"

def download_model(file_path="best_model.pth"):
    """
    Download the model file if it does not exist locally.
    Args:
        file_path: Path to save the model file.
    """
    if not os.path.exists(file_path):
        try:
            print("Downloading model... This may take a while.")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            # Save the file in chunks
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            print("Model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the model: {e}")
            raise RuntimeError("Failed to download the model file. Please check the URL or your internet connection.")
    else:
        print("Model file already exists locally.")

# Download the model before loading it
download_model()

# Ensure the model file exists before proceeding
if not os.path.exists("best_model.pth"):
    raise FileNotFoundError("Model file 'best_model.pth' not found. Download may have failed.")

try:
    # Initialize the model
    print("Initializing the model...")
    model = ResNet18Classifier(num_classes=6, input_channels=3)

    # Load the state dictionary
    state_dict = torch.load("best_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    print("The trained deep learning model has been initialized successfully.")
except Exception as e:
    print(f"Error during model initialization: {e}")
    raise SystemExit("Model initialization failed. Exiting application.")


st.write("Upload an image, and the model will predict!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.write("Image uploaded successfully.")
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        # st.write("Image displayed successfully.")

        # Convert the image to a NumPy array
        # st.write("Converting the image to NumPy format...")
        image_np = np.array(image)  # PIL image to numpy array
        # st.write(f"Image shape (NumPy): {image_np.shape}")

        # Handle grayscale images
        if len(image_np.shape) == 2:  # If grayscale, convert to 3 channels
            # st.write("Converting grayscale image to 3 channels...")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # Convert RGB to BGR for OpenCV compatibility
        # st.write("Converting RGB image to BGR format...")
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # st.write("Image converted to OpenCV BGR format.")

        # Preprocess the image using the custom pipeline
        # st.write("Starting image preprocessing...")
        input_tensor = preprocess_image(image_cv).unsqueeze(0)  # Add batch dimension
        # st.write("Image preprocessing completed successfully.")

        # Prediction
        st.write("Starting model prediction...")
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = prediction.argmax(dim=1).item()
        st.write(f"Predicted Class: {predicted_class+1}")
    except Exception as e:
        st.error(f"Error during preprocessing or prediction: {e}")
        st.stop()
