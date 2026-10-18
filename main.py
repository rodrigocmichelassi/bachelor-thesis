import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import torch
from src.data.import_data import import_data
from src.training.train import train_classifier
from src.training.evaluate import evaluate_classifier, evaluate_retrieval, calculate_classification_metrics, show_retrieval_metrics
from src.utils.helper import log_zero_shot_results, plot_training_evolution, get_run_checkpoint_path, set_seed
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
def run_zero_shot(args, test_loader, log_results=False):
    print(f"Running zero-shot {args.zero_shot_mode}")
    
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model, processor = load_raw_clip_model()

    if args.zero_shot_mode == 'classifier':
        class_labels = build_class_labels()

        true_labels, pred_labels, loss = evaluate_classifier(model, processor, test_loader, class_labels, device)
        acc, bal_acc, report, cm = calculate_classification_metrics(true_labels, pred_labels, class_labels)

        if log_results:
            log_zero_shot_results(acc, loss, bal_acc, cm, report)
        
    elif args.zero_shot_mode == 'retrieval':
        recall_at_k = evaluate_retrieval(model, test_loader, device)
        show_retrieval_metrics(recall_at_k)
    
    else:
        raise ValueError(f"Invalid zero-shot evaluation mode!")

# train_classifier_model train a CLIP classifier using the captions under
# `data/classification_captions.csv` for each BRSET image
def train_classifier_model(args, train_loader, val_loader, class_labels):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model, processor = load_lora_clip(debug=True)
    train_acc, train_loss, val_acc, val_loss = train_classifier(args, device, model, train_loader, val_loader, processor, class_labels)

    plot_training_evolution(args, train_acc, train_loss, val_acc, val_loss)

# evaluate_classifier_model evaluates the results of a finetuned model for classification
def evaluate_classifier_model(args, test_loader, class_labels):
    print(f"Evaluate model with lr: {args.lr}, l2: {args.l2}")

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_path = get_run_checkpoint_path(args)

    model, processor = load_fine_tuned_clip(checkpoint_path)
    true_labels, pred_labels, _ = evaluate_classifier(model, processor, test_loader, class_labels, device, debug=False)
    calculate_classification_metrics(true_labels, pred_labels, class_labels)

def evaluate_retrieval_model(args, test_loader):
    print(f"Evaluate model with lr: {args.lr}, l2: {args.l2}")
    
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_path = get_run_checkpoint_path(args)

def main(args):
    set_seed(args.seed)
    train_loader, val_loader, test_loader = import_data(args)
    
    if args.training_mode == 'zero_shot':
        run_zero_shot(args, test_loader, log_results=True)
        return 0

    print(f"Training CLIP with lr: {args.lr} - l2: {args.l2} - epochs: {args.epochs}")
    if args.training_mode == 'classifier':
        class_labels = build_class_labels()
        train_classifier_model(args, train_loader, val_loader, class_labels)
        evaluate_classifier_model(args, test_loader, class_labels)

    if args.training_mode == 'retrieval':
        pass

    if args.evaluate_classifier:
        evaluate_classifier_model(args, test_loader, class_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Operations with CLIP model.")

    parser.add_argument('--seed', type=int, default=42, help='Seed value for training reproducibility')
    parser.add_argument('--gpu', type=int, default=0, help='Index of the GPU to be used for training')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Training Learning Rate')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization value')
    parser.add_argument('--weight_classes', type=int, default=0, help='0: no weighting, 1: inverse frequency, 2: sqrt-smoothed inverse frequency')
    parser.add_argument('--training_mode', type=str, default='retrieval', help='One of: {zero_shot, retrieval, classifier}')
    parser.add_argument('--zero_shot_mode', type=str, default='retrieval', help='One of: {classifier, retrieval}')
    parser.add_argument('--retrieval_mode', type=str, default='single_row', help='Which dataset to use for image retrieval training. One of: {single_row, multi_row}')
    parser.add_argument('--evaluate_classifier', type=int, default=0, help='Run *only* evaluation, model chosen based on lr and l2 weights')

    args = parser.parse_args()

    main(args)