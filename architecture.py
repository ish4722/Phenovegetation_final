import torch.nn as nn
import torch.nn.functional as F
from config import *


# Define the model using PyTorch
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=n_class):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fix: Final convolution layer should output 'n_classes' channels
        self.final_conv = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.final_conv(x)  # Ensure final shape (batch_size, n_classes, H, W)
        return x