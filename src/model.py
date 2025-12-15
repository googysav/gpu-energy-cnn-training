## Imports
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

## Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

## Googly drive access (for saving results)
from google.colab import drive
drive.mount("/content/drive")

RESULTS_PATH = "/content/drive/MyDrive/cifar_energy_results.json"

#
# Creates models and convolution/pooling stack
#
class SimpleCNN(nn.Module):
    """
    Small CNN for CIFAR-10 with a configurable width.
    width=1 is tiny, width=2 doubles channels... etc.
    """

    def __init__(self, num_classes: int = 10, width: int = 1):
        super().__init__()

        ## Define channel sizes
        c1 = 32 * width
        c2 = 64 * width
        c3 = 128 * width

        ## Create feature extractor
        """
        Spacial resolution decreases as feature channel
        depth increases. Network builds heirarchy of features.
        (local low level -> global high level as channel size
        increases)
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        ## Create classifier
        """
        Collapse feature maps into feature vectors/class logits
        """
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256 * width),
            nn.ReLU(inplace=True),
            nn.Linear(256 * width, num_classes),
        )

    ## Push images through convolution and pooling stack
    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
