import json
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt

def log_zero_shot_results(acc, loss, bal_acc, cm, report):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    result = {
        "run": "zero_shot_classification",
        "timestamp": timestamp,
        "accuracy": acc,
        "loss": loss,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm.tolist(),
    }

    with open(log_dir / f"zero_shot_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)

    # human-readable version alongside it
    with open(log_dir / f"zero_shot_{timestamp}.txt", "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Balanced Accuracy: {bal_acc}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

def plot_training_evolution(train_acc, train_loss, val_acc, val_loss, save_path):
    epochs = range(1, len(train_acc) + 1)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axes[0].plot(epochs, train_acc, label="Train Accuracy", marker="o")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    axes[0].set_title("Accuracy Evolution")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[1].plot(epochs, val_loss, label="Validation Loss", marker="o")
    axes[1].set_title("Loss Evolution")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

if __name__ == '__main__':
    pass