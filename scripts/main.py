## Imports
import time
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from src.data import get_cifar10_loaders
from src.experiment import run_experiment
from scripts.plots import plot_results 

## Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

RESULTS_PATH = "results/cifar_energy_results.json"
FIGURES_DIR = "figures"

#
# Entrypoint and model definitions
#
def main():
    #Model variables
    epochs = 30
    batch_size = 128
    lr = 0.1
    weight_decay = 5e-4
    gpu_tdp_watts = 70.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader, val_loader = get_cifar10_loaders(batch_size=batch_size)

    metrics_list = []

    # 1. LARGER MODEL (Width 3, fp32)
    metrics_list.append(
        run_experiment(
            config_name="fp32_width3",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs,
            width=3,
            base_lr=lr,
            weight_decay=weight_decay,
            use_amp=False,
            gpu_tdp_watts=gpu_tdp_watts,
        )
    )
    # 2. LARGER MIXED PRECISION MODEL (Width 3, Amp)

    metrics_list.append(
        run_experiment(
            config_name="amp_width3",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs,
            width=3,
            base_lr=lr,
            weight_decay=weight_decay,
            use_amp=True,
            gpu_tdp_watts=gpu_tdp_watts,
            )
        )
    # 3. BASELINE MODEL, (Width 2, fp32)
    metrics_list.append(
        run_experiment(
            config_name="fp32_width2",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs,
            width=2,
            base_lr=lr,
            weight_decay=weight_decay,
            use_amp=False,
            gpu_tdp_watts=gpu_tdp_watts,
        )
    )

    # 4. BASELINE MIXED PRECISION MODEL (Width 2, Amp)

    metrics_list.append(
        run_experiment(
            config_name="amp_width2",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs,
            width=2,
            base_lr=lr,
            weight_decay=weight_decay,
            use_amp=True,
            gpu_tdp_watts=gpu_tdp_watts,
            )
        )

    # 5. SMALLER MODEL (Width 1, fp32)
    metrics_list.append(
        run_experiment(
            config_name="fp32_width1",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs,
            width=1,
            base_lr=lr,
            weight_decay=weight_decay,
            use_amp=False,
            gpu_tdp_watts=gpu_tdp_watts,
        )
    )

    # 6. SMALLER MIXED PRECISION MODEL (Width 1, Amp)

    metrics_list.append(
        run_experiment(
            config_name="amp_width1",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs,
            width=1,
            base_lr=lr,
            weight_decay=weight_decay,
            use_amp=True,
            gpu_tdp_watts=gpu_tdp_watts,
            )
        )

    # Print summary table
    print("\nSummary:")
    header = (
        f"{'config':25s} {'best_acc':>9s} {'avg_epoch_s':>12s} "
        f"{'total_s':>10s} {'energy_kWh':>12s}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m.config_name:25s} "
            f"{m.best_val_acc:9.2f} "
            f"{m.avg_epoch_time_sec:12.2f} "
            f"{m.total_time_sec:10.2f} "
            f"{m.estimated_energy_kwh:12.5f}"
        )

    # Save to json for visualisation
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump([asdict(m) for m in metrics_list], f, indent=2)

    print("Saved results to", RESULTS_PATH)

    plot_results(RESULTS_PATH, FIGURES_DIR, show=True)
    print("Saved figures to", FIGURES_DIR)


if __name__ == "__main__":
    main()
