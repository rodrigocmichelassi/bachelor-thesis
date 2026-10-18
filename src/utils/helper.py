import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from src.config import MODELS_DIR, PLOTS_DIR, IMAGES_DIR
from src.dataset import SingleRowRetrievalDataset, MultiRowRetrievalDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_run_checkpoint_path(args, base_dir=MODELS_DIR):
    chosen_model = 'classifier' if args.training_mode == 'classifier' else 'retrieval'
 
    run_name = f"lr_{args.lr}_l2_{args.l2}_wc_{args.weight_classes}"
    return base_dir / f"clip_weights/{chosen_model}" / run_name

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
    axes[0].set_title(f"Accuracy Evolution - lr: {args.lr}, l2: {args.l2}, wc: {args.weight_classes}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[1].plot(epochs, val_loss, label="Validation Loss", marker="o")
    axes[1].set_title(f"Loss Evolution - lr: {args.lr}, l2: {args.l2}, wc: {args.weight_classes}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = PLOTS_DIR / f"wc{args.weight_classes}"
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"lr_{args.lr}_l2_{args.l2}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

# to_long_format gets the 3 caption columns and split into 3 rows,
# with "image_id" and "caption" cols
def to_long_format(df, seed=42):
    cols = ["disease_caption", "anatomy_caption", "quality_caption"]
    long_df = (
        df.melt(id_vars="image_id", value_vars=cols, value_name="caption")
          [["image_id", "caption"]]
    )
    return long_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# get_retrieval_dataset returns the dataset to be used for training
# and evaluation, based on the retrieval_mode chosen
def get_retrieval_dataset(args, train_df, test_df, val_df):
    if args.retrieval_mode == 'single_row':
        train_ds = SingleRowRetrievalDataset(train_df, IMAGES_DIR)  # random caption per epoch
        test_ds = MultiRowRetrievalDataset(to_long_format(test_df, args.seed), IMAGES_DIR)
        val_ds = MultiRowRetrievalDataset(to_long_format(val_df, args.seed), IMAGES_DIR)
    elif args.retrieval_mode == 'multi_row':
        train_ds = MultiRowRetrievalDataset(train_df, IMAGES_DIR)
        test_ds = MultiRowRetrievalDataset(test_df, IMAGES_DIR)
        val_ds = MultiRowRetrievalDataset(val_df, IMAGES_DIR)
    else:
        raise ValueError('--retrieval_mode is one of {single_row, multi_row}.')

    return train_ds, test_ds, val_ds

if __name__ == '__main__':
    pass