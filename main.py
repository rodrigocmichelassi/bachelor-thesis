import argparse
import torch
import json
from datetime import datetime
from pathlib import Path

from src.data.import_data import import_data
from src.models.model import load_raw_clip_model
from src.training.evaluate import evaluate_dataset, calculate_classification_metrics

# build_class_labels returns a list of classification class labels
def build_class_labels():
    DISEASE_COLUMNS = [
        'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
        'vascular_occlusion', 'hypertensive_retinopathy', 'drusens',
        'hemorrhage', 'myopic_fundus', 'increased_cup_disc', 'other_abnormalities'
    ]

    class_labels = {"healthy": "This is an image of a healthy retina."}

    for disease in DISEASE_COLUMNS:
        disease_str = disease.replace('_', ' ')
        class_labels[disease_str] = f"This is an image of {disease_str}."

    return class_labels

# run_zero_shot runs a zero-shot classification on the test set
def run_zero_shot(test_loader, class_labels, log_results=False):
    print("Running zero-shot classification")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, processor = load_raw_clip_model()
    true_labels, pred_labels = evaluate_dataset(model, processor, test_loader, class_labels, device)
    acc, bal_acc, report, cm = calculate_classification_metrics(true_labels, pred_labels, class_labels)

    if log_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        result = {
            "run": "zero_shot_classification",
            "timestamp": timestamp,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "confusion_matrix": cm.tolist(),
        }

        with open(log_dir / f"zero_shot_{timestamp}.json", "w") as f:
            json.dump(result, f, indent=2)

        # human-readable version alongside it
        with open(log_dir / f"zero_shot_{timestamp}.txt", "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Balanced Accuracy: {bal_acc}\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

def main(args):
    train_loader, val_loader, test_loader = import_data()
    class_labels = build_class_labels()

    if args.zero_shot:
        run_zero_shot(test_loader, class_labels, log_results=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Operations with CLIP model.")

    parser.add_argument('--zero_shot', type=int, default=0, help='Run a zero-shot classification with CLIP (no fine-tuning/LoRA)')

    args = parser.parse_args()

    # next steps:
    # 1. Run zero-shot evaluation to define baseline. Define fixated seed for testing comparison
    # 2. LoRA for fine-tuning CLIP + training loop

    main(args)