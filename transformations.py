import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Normalization values for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_southwest_quadrant(image):
    """
    Crop the southwest quadrant (bottom-left) of the image.
    Args:
        image: Input image as a numpy array (H x W x C).
    Returns:
        Cropped image (numpy array).
    """
    # print("[DEBUG] Cropping the southwest quadrant...")
    height, width = image.shape[:2]
    x_start = 0
    y_start = height // 2
    x_end = width // 2
    y_end = height
    cropped = image[y_start:y_end, x_start:x_end]
    # print(f"[DEBUG] Cropped image shape: {cropped.shape}")
    return cropped


def extract_acf_features(image):
    """
    Extract Aggregate Channel Features (ACF) from an image.
    Returns: A tuple of feature channels (numpy arrays).
    """
    print("[DEBUG] Extracting ACF features...")
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Channel 1: Red channel
    r, _, _ = cv2.split(image)

    # Channel 2: Gradient Magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # print(f"[DEBUG] Extracted channels: Red channel shape: {r.shape}, Gradient magnitude shape: {gradient_magnitude.shape}")
    return r, gradient_magnitude


def resize_with_aspect_ratio(image, target_size=(224, 224)):
    """
    Resize an image to the target size while maintaining its aspect ratio.
    Args:
        image: Input image as a numpy array (H x W x C).
        target_size: Desired size as (height, width).
    Returns:
        Resized and padded image (numpy array).
    """
    # print("[DEBUG] Resizing image while maintaining aspect ratio...")
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 1:  # Wider than tall
        new_w = target_size[1]
        new_h = int(new_w / aspect_ratio)
    else:  # Taller than wide
        new_h = target_size[0]
        new_w = int(new_h * aspect_ratio)

    resized = cv2.resize(image, (new_w, new_h))
    # print(f"[DEBUG] Resized image shape: {resized.shape}")

    # Pad the resized image to make it exactly target_size
    pad_top = (target_size[0] - new_h) // 2
    pad_bottom = target_size[0] - new_h - pad_top
    pad_left = (target_size[1] - new_w) // 2
    pad_right = target_size[1] - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    print(f"[DEBUG] Padded image shape: {padded.shape}")
    return padded


def get_transforms():
    """
    Define data augmentation and normalization transforms for training and testing.
    Returns:
        torchvision.transforms.Compose object for training and testing.
    """
    # print("[DEBUG] Creating data augmentation and normalization transforms...")
    return T.Compose([
        T.ToPILImage(),  # Convert numpy array to PIL image
        T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        T.RandomVerticalFlip(p=0.5),  # Random vertical flip
        T.RandomRotation(degrees=30),  # Random rotation
        T.Resize((224, 224)),  # Resize to target size
        T.ToTensor(),  # Convert PIL image to tensor
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Normalize with ImageNet stats
    ])


def preprocess_image(image):
    """
    Complete preprocessing pipeline for an image.
    Args:
        image: Input image as a numpy array (H x W x C) in BGR format.
    Returns:
        Preprocessed image tensor ready for deep learning models.
    """
    # print("[DEBUG] Starting preprocessing pipeline...")

    # Check if image is grayscale and convert to 3 channels if necessary
    if len(image.shape) == 2:
        print("[DEBUG] Converting grayscale image to 3 channels...")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # print("[DEBUG] Original image shape:", image.shape)

    # Crop the southwest quadrant
    cropped_image = crop_southwest_quadrant(image)

    # Extract meaningful features
    channel_1, channel_2 = extract_acf_features(cropped_image)

    # Resize channels
    resized_1 = resize_with_aspect_ratio(channel_1, target_size=(224, 224))
    resized_2 = resize_with_aspect_ratio(channel_2, target_size=(224, 224))

    # Combine the channels into a single 3-channel image
    # print("[DEBUG] Stacking resized channels into a 3-channel image...")
    three_channel_image = np.stack([resized_1, resized_2, resized_2], axis=-1)
    print(f"[DEBUG] Final stacked image shape: {three_channel_image.shape}")

    # Apply data augmentation and normalization
    # print("[DEBUG] Applying data augmentation and normalization...")
    transforms = get_transforms()
    transformed_image = transforms(three_channel_image)

    # print("[DEBUG] Preprocessing completed successfully.")
    return transformed_image


# Exported functions for external use
__all__ = [
    "crop_southwest_quadrant",
    "extract_acf_features",
    "resize_with_aspect_ratio",
    "get_transforms",
    "preprocess_image",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
