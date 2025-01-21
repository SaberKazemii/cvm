#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# # Load the original full image
# im_dir = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/output.csv"
# # Define the directory containing the images
im_dir = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/saber kokey"

# # Loop through all image IDs
# for i in range(1, 1229):
#     im_id = f"{i}.jpg"  # Construct the filename
#     full_im_addr = os.path.join(im_dir, im_id)

#     # Load the full image
#     full_image = cv2.imread(full_im_addr)
#     if full_image is None:
#         print(f"Image {im_id} not found, skipping...")
#         continue  # Skip if the image doesn't exist

#     # Get the dimensions of the full image
#     height, width = full_image.shape[:2]
#     print(f"Processing {im_id}: Height={height}, Width={width}")

#     # Define the southwest quadrant (bottom-left)
#     x_start = 0
#     y_start = height // 2
#     x_end = width // 2
#     y_end = height

#     # Crop the southwest quadrant
#     sw_quadrant = full_image[y_start:y_end, x_start:x_end]

#     # Convert to grayscale
#     gray_sw = cv2.cvtColor(sw_quadrant, cv2.COLOR_BGR2GRAY)

#     # Perform binary segmentation using thresholding
#     # You can adjust the threshold value (e.g., 127) based on your needs
#     _, binary_segmented = cv2.threshold(gray_sw, 40, 255, cv2.THRESH_BINARY)

#     # Show the segmented image
#     plt.figure(figsize=(12, 6))

#     # Original cropped quadrant
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(sw_quadrant, cv2.COLOR_BGR2RGB))
#     plt.title("Southwest Quadrant (Original)")
#     plt.axis("off")

#     # Binary segmented quadrant
#     plt.subplot(1, 2, 2)
#     plt.imshow(binary_segmented, cmap="gray")
#     plt.title("Binary Segmentation")
#     plt.axis("off")

#     # Show the results
#     plt.show()

#     # Save the segmented image (optional)
#     segmented_filename = f"segmented_{i}.jpg"
#     cv2.imwrite(segmented_filename, binary_segmented)
#     print(f"Segmented image saved as {segmented_filename}")

#     # Convert the southwest quadrant to grayscale
# gray_sw_quadrant = cv2.cvtColor(sw_quadrant, cv2.COLOR_BGR2GRAY)

# # Load the target template (the cropped part)
# template = cv2.imread('template.jpg', 0)  # Replace with the filename of the template part
# w, h = template.shape[::-1]

# # Match the template in the southwest quadrant
# res = cv2.matchTemplate(gray_sw_quadrant, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8 # Adjust the threshold as needed
# loc = np.where(res >= threshold)

# # Annotate and crop the detected area
# for pt in zip(*loc[::-1]):  # Switch x and y coordinates
#     # Adjust the coordinates to the original image
#     top_left = (pt[0] + x_start, pt[1] + y_start)
#     bottom_right = (top_left[0] + w, top_left[1] + h)

#     # Draw rectangle for visualization
#     cv2.rectangle(full_image, top_left, bottom_right, (255, 0, 0), 2)

#     # Crop the matched region from the original image
#     cropped_part = full_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
#     cv2.imwrite('cropped_output.jpg', cropped_part)  # Save the cropped part
#     print("Cropped part saved as 'cropped_output.jpg'")
#     break  # Only process the fir



# In[ ]:


import torch

# Assign the device based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the assigned device
print(f"Using device: {device}")

# If using a GPU,


# In[ ]:


import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
import os

def extract_acf_features(image):
    """
    Extract Aggregate Channel Features (ACF) from an image.
    Returns: A list of feature channels (numpy arrays).
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Channel 1: RGB channels
    r, g, b = cv2.split(image)
    
    # Channel 5: Gradient Magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    return r, gradient_magnitude  # Return only Channel 1 and Channel 5


def crop_and_resize(image, target_size=(224, 224)):
    """
    Crop the southwest quadrant of the image and resize it to the target size
    while maintaining the aspect ratio.
    """
    # Get dimensions of the image
    height, width = image.shape[:2]
    
    # Define the southwest quadrant (bottom-left)
    x_start = 0
    y_start = height // 2
    x_end = width // 2
    y_end = height

    # Crop the southwest quadrant
    cropped = image[y_start:y_end, x_start:x_end]

    # Calculate aspect ratio
    h, w = cropped.shape[:2]
    aspect_ratio = w / h

    # Resize while maintaining aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_w = target_size[1]
        new_h = int(new_w / aspect_ratio)
    else:  # Taller than wide
        new_h = target_size[0]
        new_w = int(new_h * aspect_ratio)

    resized = cv2.resize(cropped, (new_w, new_h))

    # Pad the resized image to make it exactly target_size
    pad_top = (target_size[0] - new_h) // 2
    pad_bottom = target_size[0] - new_h - pad_top
    pad_left = (target_size[1] - new_w) // 2
    pad_right = target_size[1] - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Black padding
    )

    return padded


# Main Script
im_dir = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/saber kokey/saber kokey"
im_id = f"{248}.jpg"  # Replace with your image index
full_im_addr = os.path.join(im_dir, im_id)

# Load the full image
full_image = cv2.imread(full_im_addr)
if full_image is None:
    print(f"Image not found: {full_im_addr}")
    exit()

# Extract Channel 1 and Channel 5
channel_1, channel_5 = extract_acf_features(full_image)

# Crop and resize Channel 1 and Channel 5
cropped_resized_1 = crop_and_resize(channel_1)
cropped_resized_5 = crop_and_resize(channel_5)

# # Plot the results
# plt.figure(figsize=(10, 5))

# # Channel 1 (Red)
# plt.subplot(1, 2, 1)
# plt.imshow(cropped_resized_1, cmap="gray")
# plt.title("Channel 1: Red (Cropped and Resized)")
# plt.axis("off")

# # Channel 5 (Gradient Magnitude)
# plt.subplot(1, 2, 2)
# plt.imshow(cropped_resized_5, cmap="gray")
# plt.title("Channel 5: Gradient Magnitude (Cropped and Resized)")
# plt.axis("off")

# plt.show()


# In[3]:


import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as T
import os


class XRayDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, target_size=(224, 224), valid_extensions=None):
        """
        Dataset constructor.
        Args:
            csv_path: Path to the CSV file (first column: image_id, second column: label).
            image_dir: Directory containing the images.
            transform: Optional torchvision transforms for data augmentation.
            target_size: Tuple (height, width) for resizing.
            valid_extensions: List of valid file extensions to check for images.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.valid_extensions = valid_extensions or ['.jpg', '.png', '.tif', '.jpeg']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image ID and label from the CSV
        row = self.data.iloc[idx]
        image_id = str(int(row[0]))  # Ensure image ID is a string
        # print(image_id, str(row[1][-1]))
        label = int(str(row[1][-1]))-1         # Second column: label

        # Find the image file with valid extension
        image_path = self.find_image_file(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for ID {image_id} in supported formats.")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Crop the southwest quadrant
        cropped_image = self.crop_southwest_quadrant(image)

        # Extract 3 meaningful channels
        channel_1, channel_2, channel_3 = self.extract_features(cropped_image)

        # Resize all channels and stack them into a 3-channel image
        resized_1 = self.resize_with_aspect_ratio(channel_1, self.target_size)
        resized_2 = self.resize_with_aspect_ratio(channel_2, self.target_size)
        resized_3 = self.resize_with_aspect_ratio(channel_3, self.target_size)

        three_channel_image = np.stack([resized_1, resized_2, resized_3], axis=-1)

        # Apply data augmentation (if transform is provided)
        if self.transform:
            three_channel_image = self.apply_transforms(three_channel_image)

        # Convert to tensor and return
        return (
            torch.tensor(three_channel_image, dtype=torch.float32).permute(2, 0, 1),  # Channels first
            torch.tensor(int(label), dtype=torch.long)             # Label
        )

    def find_image_file(self, image_id):
        """
        Search for an image file with a valid extension for the given image ID.
        Args:
            image_id: The base name of the image file (without extension).
        Returns:
            Full path to the image file if found, otherwise None.
        """
        for ext in self.valid_extensions:
            image_path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        return None

    def crop_southwest_quadrant(self, image):
        """
        Crop the southwest quadrant (bottom-left) of the image.
        """
        height, width = image.shape[:2]
        x_start = 0
        y_start = height // 2
        x_end = width // 2
        y_end = height
        return image[y_start:y_end, x_start:x_end]

    def extract_features(self, image):
        """
        Extract 3 meaningful channels: Grayscale, Gradient Magnitude, Laplacian of Gaussian.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Channel 1: Grayscale
        channel_1 = gray

        # Channel 2: Gradient Magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        channel_2 = cv2.convertScaleAbs(gradient_magnitude)

        # Channel 3: Laplacian of Gaussian (LoG)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        channel_3 = cv2.convertScaleAbs(laplacian)

        return channel_1, channel_2, channel_3

    def resize_with_aspect_ratio(self, image, target_size):
        """
        Resize an image to the target size while maintaining its aspect ratio.
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h

        if aspect_ratio > 1:  # Wider than tall
            new_w = target_size[1]
            new_h = int(new_w / aspect_ratio)
        else:  # Taller than wide
            new_h = target_size[0]
            new_w = int(new_h * aspect_ratio)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad the resized image to make it exactly target_size
        pad_top = (target_size[0] - new_h) // 2
        pad_bottom = target_size[0] - new_h - pad_top
        pad_left = (target_size[1] - new_w) // 2
        pad_right = target_size[1] - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded

    def apply_transforms(self, image):
        """
        Apply torchvision transformations to a 3-channel image.
        """
        # Convert numpy image to PIL image for compatibility with torchvision
        image = T.ToPILImage()(image)



        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            T.RandomVerticalFlip(p=0.5),    # Random vertical flip
            T.RandomRotation(degrees=30),   # Random rotation
            T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random crop and resize
            T.ToTensor(),                   # Convert image to PyTorch tensor
            T.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize using ImageNet mean and std
        ])
        augmented = transform(image)

        # Convert back to numpy array
        return augmented.permute(1, 2, 0).numpy()
        # Apply random augmentations
        # Normalization values for ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]



# In[4]:


import matplotlib.pyplot as plt

# # Define paths
# csv_path = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/output.csv"  # CSV file containing image IDs and labels
# image_dir = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/saber kokey/saber kokey"

# # Initialize dataset
# dataset = XRayDataset(csv_path, image_dir, transform=True)

# index = 0  # Index of the sample in the dataset

# # Number of augmented samples to generate
# num_augmentations = 6  

# # Fetch the original sample and repeatedly apply augmentation
# fig, axes = plt.subplots(num_augmentations, 1, figsize=(8, num_augmentations * 4))

# for i in range(num_augmentations):
#     # Fetch an augmented version of the sample
#     input_tensor, label = dataset[index]  # Unpack the tuple
#     print(input_tensor.shape)  # Check the shape of the input tensor

#     # Plot the input tensor
#     axes[i].imshow(input_tensor.squeeze(), cmap="gray")  # Squeeze to remove channel dim
#     axes[i].set_title(f"Augmented Sample (Label: {label.item()})")
#     axes[i].axis("off")

# plt.tight_layout()
# plt.show()


# In[5]:


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pandas as pd

# Define file paths
csv_path = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/output.csv"  # CSV file path
image_dir = "/shared/hdds_20T/sk1017/Dropbox (Partners HealthCare)/Usef's Project/dataset/saber kokey/saber kokey"  # Image directory

# Initialize the dataset
dataset = XRayDataset(csv_path, image_dir, transform=True)

# Split indices for train/val/test (80/10/10)
dataset_size = len(dataset)
indices = list(range(dataset_size))
train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# Subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Calculate weights for the train dataset
train_labels = pd.Series([dataset.data.iloc[idx, 1] for idx in train_indices])  # Extract labels
train_labels = train_labels.apply(lambda x: int(str(x)[-1]) - 1)  # Convert to class indices
class_counts = train_labels.value_counts()
total_samples = len(train_labels)
class_weights = total_samples / (len(class_counts) * class_counts)  # Compute class weights
sample_weights = train_labels.map(class_weights).values  # Map weights to samples

# Create a WeightedRandomSampler for the train dataset
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

# Create data loaders
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)  # Use sampler
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset sizes for verification
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")


# In[6]:


import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        
        # Load pre-trained ResNet-18 model
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 2 input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels=2,  # Two input channels: Channel 1 and Channel 5
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        
        # Freeze all the layers of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the classification head with a custom head
        # Input features: 512 (output of ResNet-18 backbone)
        self.base_model.fc = nn.Sequential(
            nn.Linear(512, 256),  # Reduce features from 512 to 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Final layer for 6-class output
        )
        
        # Count trainable parameters for verification
        self.count_trainable_params()

    def count_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    def forward(self, x):
        return self.base_model(x)


# In[7]:


import torch
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6, input_channels=3, pretrained_weights_path=None):
        """
        ResNet18-based classifier with a customizable classification head.
        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels (e.g., 3 for RGB or feature channels).
            pretrained_weights_path: Path to the pre-trained weights file (optional).
        """
        super(ResNet18Classifier, self).__init__()
        
        # Load pre-trained ResNet18 model
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all layers of the base model except the last layer (layer4)
        for name, param in self.base_model.named_parameters():
            if not name.startswith("layer4"):  # Freeze everything except "layer4"
                param.requires_grad = False
        
        # Modify the first convolutional layer to accept custom input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels=input_channels,  # Match the number of input channels
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.base_model.conv1.requires_grad = True  # Keep the modified conv1 layer trainable
        
        # Add a new classification head with additional layers
        self.base_model.fc = nn.Sequential(
            nn.Linear(512, 256),      # First fully connected layer
            nn.ReLU(),                # Activation
            nn.Dropout(0.2),          # Dropout for regularization
            nn.Linear(256, 128),      # Additional fully connected layer
            nn.ReLU(),                # Activation
            nn.Dropout(0.2),          # Dropout for regularization
            nn.Linear(128, num_classes)  # Final layer for classification
        )

        # Load pre-trained weights if provided
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, weights_path):
        """
        Load pre-trained weights into the model.
        Args:
            weights_path: Path to the pre-trained weights file.
        """
        try:
            # Load the weights
            pretrained_weights = torch.load(weights_path, map_location=torch.device('cpu'))
            self.load_state_dict(pretrained_weights)
            print(f"Successfully loaded pre-trained weights from {weights_path}.")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        return self.base_model(x)


# In[ ]:





# In[8]:


import os
print(os.getcwd())


# In[ ]:


from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import f1_score
best_model_path = "best_model.pth"
num_classes=6
# Calculate class weights
data=pd.read_csv(csv_path)
all_labels = data['labels'].tolist()
# Get the unique labels (classes) from all_labels
unique_classes = np.unique(all_labels)
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
# Convert class weights to a tensor for PyTorch
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class weights:", class_weights_tensor)
# Parameters for model initialization
num_classes = 6  # Number of classes for classification
input_channels = 3  # Number of input channels (e.g., 3 for RGB images)
pretrained_weights_path = None  # Path to pre-trained weights file (set to None if not using)

# Create an instance of the ResNet18Classifier
model = ResNet18Classifier(
    num_classes=num_classes,
    input_channels=input_channels,
    pretrained_weights_path=pretrained_weights_path
)
model=model.to(device)
# Print the model structure to verify
print(model)
# Define the weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Add weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Data augmentation
from torchvision import transforms
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

# Modify the dataloaders to include augmentation for the training set
train_dataset.transform = data_augmentation

# Revised Training Loop with Balanced Metrics
num_epochs = 100
best_val_f1 = 0.0  # Track the best F1-score

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in train_loader_tqdm:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate balanced F1-score
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total * 100
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Validation F1-Score: {val_f1:.4f}")

    # Save the best model based on validation F1-score
    if val_f1 > best_val_f1:
        print(f"Validation F1-Score improved from {best_val_f1:.4f} to {val_f1:.4f}. Saving model...")
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)

print(f"Training complete. Best model saved to {best_model_path}.")


# In[10]:


import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set the model to evaluation mode

# Initialize placeholders
all_labels = []
all_preds = []
all_probs = []

# Test phase
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)  # Get probabilities
        preds = torch.argmax(probs, dim=1)  # Get predictions

        # Collect predictions and labels
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy arrays for metric calculations
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(6), yticklabels=range(6))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))

# ROC-AUC and ROC Curve
plt.figure(figsize=(10, 8))
fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC for each class
n_classes = all_probs.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

# Micro-average ROC-AUC
fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-Average (AUC = {roc_auc['micro']:.2f})", linestyle='--')

# Macro-average ROC-AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
roc_auc["macro"] = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, label=f"Macro-Average (AUC = {roc_auc['macro']:.2f})", linestyle=':')

# Plot formatting
plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Classes")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Print Macro and Micro AUC
print(f"Micro-Average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-Average AUC: {roc_auc['macro']:.4f}")

