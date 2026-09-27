import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from src.config import MODELS_DIR, PLOTS_DIR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_run_checkpoint_path(lr, l2, base_dir=MODELS_DIR):
    run_name = f"lr_{lr}_l2_{l2}"
    return base_dir / "clip_weights" / run_name

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

def plot_training_evolution(args, train_acc, train_loss, val_acc, val_loss):
    save_path = PLOTS_DIR
    epochs = range(1, len(train_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axes[0].plot(epochs, train_acc, label="Train Accuracy", marker="o")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    axes[0].set_title(f"Accuracy Evolution - lr: {args.lr}, l2: {args.l2}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[1].plot(epochs, val_loss, label="Validation Loss", marker="o")
    axes[1].set_title(f"Loss Evolution - lr: {args.lr}, l2: {args.l2}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / f"lr_{args.lr}_l2_{args.l2}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

if __name__ == '__main__':
    pass