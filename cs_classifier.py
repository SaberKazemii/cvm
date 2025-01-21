#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6, input_channels=3):
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




