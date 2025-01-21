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
