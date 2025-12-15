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

from src.model import SimpleCNN
from src.metrics import RunMetrics
from src.train_eval import train_one_epoch, evaluate

## Googly drive access (for saving results)
from google.colab import drive
drive.mount("/content/drive")

RESULTS_PATH = "/content/drive/MyDrive/cifar_energy_results.json"

#
# Main experiment runner
#
def run_experiment(
    config_name: str,
    train_loader,
    val_loader,
    device,
    num_epochs: int,
    width: int,
    base_lr: float,
    weight_decay: float,
    use_amp: bool,
    gpu_tdp_watts: float,
) -> RunMetrics:

    """
    Create heap objects for the model, loss function,
    optimizer, and gradient scalar
    """
    model = SimpleCNN(num_classes=10, width=width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    total_start = time.time() #Start time for total model runtime

    #Initialise tracking variables/metrics
    best_val_acc = 0.0
    epoch_times = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    estimated_energies = []


    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time() #Start time for epoch runtime
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                device, use_amp=use_amp, scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        #Update metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_end = time.time() #End time for epoch runtime

        #Compute epoch runtime and track
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        #Update best validation accuracy
        best_val_acc = max(best_val_acc, val_acc)

        #Compute energy for this epoch (cumulative)
        epoch_energy = gpu_tdp_watts * (epoch_time/3600.0) / 1000
        if len(estimated_energies) == 0:
          estimated_energies.append(epoch_energy)
        else:
          estimated_energies.append(estimated_energies[-1] + epoch_energy)

        #print data for epoch
        print(
            f"[{config_name}] Epoch {epoch:02d}/{num_epochs} "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.2f}% "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2f}% "
            f"Time: {epoch_time:.2f} s"
        )

    total_end = time.time() #End time for total model runtime
    total_time = total_end - total_start #Calculate model runtime
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Rough energy estimate:
    # energy (kWh) approx = power (W) * time (h) / 1000
    total_time_hours = total_time / 3600.0
    estimated_energy_kwh = gpu_tdp_watts * total_time_hours / 1000.0


    metrics = RunMetrics(
        config_name=config_name,
        epochs=num_epochs,
        total_time_sec=total_time,
        avg_epoch_time_sec=avg_epoch_time,
        best_val_acc=best_val_acc,
        estimated_energy_kwh=estimated_energy_kwh,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accs=val_accs,
        train_accs=train_accs,
        estimated_energies=estimated_energies,
    )


    print(f"\n[{config_name}] Finished")
    print(metrics)
    print()

    return metrics
