import argparse
import torch
from src.data.import_data import import_data
from src.training.train import train_classifier
from src.training.evaluate import evaluate_dataset, calculate_classification_metrics
from src.utils.helper import log_zero_shot_results, plot_training_evolution, get_run_checkpoint_path
from src.models.model import load_raw_clip_model, load_lora_clip, load_fine_tuned_clip

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
def run_zero_shot(gpu, test_loader, class_labels, log_results=False):
    print("Running zero-shot classification")

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")

    model, processor = load_raw_clip_model()
    true_labels, pred_labels, loss = evaluate_dataset(model, processor, test_loader, class_labels, device)
    acc, bal_acc, report, cm = calculate_classification_metrics(true_labels, pred_labels, class_labels)

    if log_results:
        log_zero_shot_results(acc, loss, bal_acc, cm, report)

# train a CLIP classifier using the captions under
# `data/classification_captions.csv` for each BRSET image
def train_classifier_model(args, train_loader, val_loader, class_labels):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model, processor = load_lora_clip(debug=True)
    train_acc, train_loss, val_acc, val_loss = train_classifier(args, device, model, train_loader, val_loader, processor, class_labels)

    plot_training_evolution(args, train_acc, train_loss, val_acc, val_loss)

# Evaluate the results of a finetuned model for classification
def evaluate_classifier_model(args, test_loader, class_labels):
    print(f"Evaluate model with lr: {args.lr}, l2: {args.l2}")

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_path = get_run_checkpoint_path(args.lr, args.l2)

    model, processor = load_fine_tuned_clip(checkpoint_path)
    true_labels, pred_labels, loss = evaluate_dataset(model, processor, test_loader, class_labels, device, debug=False)
    acc, bal_acc, report, cm = calculate_classification_metrics(true_labels, pred_labels, class_labels)

    print(f"Test Results:\nAcc: {acc} - Balanced Accuracy: {bal_acc} - Loss: {loss}")
    print(report)
    print(cm)

def main(args):
    print(f"Training CLIP with lr: {args.lr} - l2: {args.l2} - epochs: {args.epochs}")

    train_loader, val_loader, test_loader = import_data()
    class_labels = build_class_labels()

    if args.zero_shot:
        run_zero_shot(args.gpu, test_loader, class_labels, log_results=True)

    if args.train_classifier:
        train_classifier_model(args, train_loader, val_loader, class_labels)
        evaluate_classifier_model(args, test_loader, class_labels)

    if args.evaluate_classifier:
        evaluate_classifier_model(args, test_loader, class_labels)

    # Next steps: 
    # - apply seed to all training, to make it reproducible
    # - train the image retrieval model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Operations with CLIP model.")

    parser.add_argument('--gpu', type=int, default=0, help='Index of the GPU to be used for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Training Learning Rate')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization value')
    parser.add_argument('--save_plots_path', type=str, help='Where to save plots')
    parser.add_argument('--zero_shot', type=int, default=0, help='Run a zero-shot classification with CLIP (no fine-tuning/LoRA)')
    parser.add_argument('--train_classifier', type=int, default=1, help='Run the training loop for CLIP classification (fine-tuning with LoRA)')
    parser.add_argument('--evaluate_classifier', type=int, default=0, help='Run only evaluation, model chosen based on lr and l2 weights')

    args = parser.parse_args()

    main(args)