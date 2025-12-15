import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_results(results_path: str, save_dir: str, show: bool = False) -> None:
    results_path = Path(results_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open("r") as f:
        results = json.load(f)

    epochs = range(1, results[0]["epochs"] + 1)

    # Training loss
    plt.figure()
    for m in results:
        plt.plot(epochs, m["train_losses"], label=m["config_name"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "training_loss.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Validation loss
    plt.figure()
    for m in results:
        plt.plot(epochs, m["val_losses"], label=m["config_name"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "validation_loss.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Training accuracy
    plt.figure()
    for m in results:
        plt.plot(epochs, m["train_accs"], label=m["config_name"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "training_accuracy.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Validation accuracy
    plt.figure()
    for m in results:
        plt.plot(epochs, m["val_accs"], label=m["config_name"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "validation_accuracy.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Cumulative energy
    plt.figure()
    for m in results:
        plt.plot(epochs, m["estimated_energies"], label=m["config_name"])
    plt.xlabel("Epoch")
    plt.ylabel("Energy Expended (kWh)")
    plt.title("Cumulative Energy Expended")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "energy_cumulative.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Energy gradients (Wh per epoch)
    gradients = []
    names = []
    for m in results:
        e = m["estimated_energies"]
        gradients.append((e[-1] - e[0]) / (epochs[-1] - epochs[0]) * 1000.0)
        names.append(m["config_name"])

    plt.figure()
    plt.bar(names, gradients)
    plt.xlabel("Models")
    plt.ylabel("Energy per Epoch (Wh)")
    plt.xticks(rotation=45)
    plt.ylim(0.38, 0.44)
    plt.title("Energy Consumption Rate per Model")
    plt.grid(True, axis="y")
    plt.savefig(save_dir / "energy_gradients.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
