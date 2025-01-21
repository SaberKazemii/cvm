import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=128, input_channels=3):
        """
        ResNet18-based classifier with an architecture that matches the saved weights.
        Args:
            num_classes: Number of output classes (matches the saved model).
            input_channels: Number of input channels (matches the saved model).
        """
        super(ResNet18Classifier, self).__init__()

        # Load pre-trained ResNet18 model
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept the correct number of input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels=input_channels,  # Match the number of input channels
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace the fully connected layer with a custom classification head
        self.base_model.fc = nn.Sequential(
            nn.Linear(512, 256),  # Intermediate fully connected layer
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(256, num_classes)  # Final classification layer (matches saved weights)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        return self.base_model(x)
