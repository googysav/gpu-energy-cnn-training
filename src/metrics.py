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
# Metrics dataclass for centralised variables
#
@dataclass
class RunMetrics:
    """
    Store all performance measurements for
    accuracy, loss, and energy. Track relevant graphical
    data in list.
    """
    config_name: str
    epochs: int
    total_time_sec: float
    avg_epoch_time_sec: float
    best_val_acc: float
    estimated_energy_kwh: float
    train_losses: list
    val_losses: list
    train_accs: list
    val_accs: list
    estimated_energies: list

