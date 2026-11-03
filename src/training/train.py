import torch
import torch.nn.functional as F
import numpy as np
# from tqdm import tqdm
from src.training.loss import contrastive_loss
from src.training.evaluate import evaluate_classifier, evaluate_retrieval
from src.utils.helper import get_run_checkpoint_path

def val_t2i_mean_recall(recall):
    t2i = recall["t2i"]
    return sum(t2i.values()) / len(t2i)

def train_classifier(args, device, model, train_loader, val_loader, processor, class_labels):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    max_acc = 0.0
    chosen_loss = 0.0

    patience = 7
    epochs_without_improvement = 0
    checkpoint_path = get_run_checkpoint_path(args)

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(args.epochs):
        contrastive_train_loss = train_one_epoch(device, model, optimizer, train_loader)

        model.eval()
        train_true_labels, train_pred_labels, train_loss = evaluate_classifier(model, processor, train_loader, class_labels, device)
        val_true_labels, val_pred_labels, val_loss = evaluate_classifier(model, processor, val_loader, class_labels, device)

        train_acc = np.mean(np.array(train_true_labels) == np.array(train_pred_labels))
        val_acc = np.mean(np.array(val_true_labels) == np.array(val_pred_labels))

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - \
                contrastive_train_loss: {contrastive_train_loss:.4f} - \
                train_loss: {train_loss:.4f} - \
                val_loss: {val_loss:.4f} - \
                train_acc: {train_acc:.4f} - \
                val_acc: {val_acc:.4f}")

        if val_acc > max_acc:
            max_acc = val_acc
            chosen_loss = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                print(f"Early stopping at epoch: {epoch}")
                break

    print(f"Chosen model - val_acc: {max_acc} - val_loss: {chosen_loss}")

    return train_acc_list, train_loss_list, val_acc_list, val_loss_list

def train_retrieval(args, device, model, train_loader, val_loader):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_t2i_recall = -1.0
    best_recall = None
    patience = 7
    epochs_without_improvement = 0
    checkpoint_path = get_run_checkpoint_path(args)

    train_loss_list, val_loss_list, val_recall_list = [], [], []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(device, model, optimizer, train_loader)

        val_recall, val_loss = evaluate_retrieval(model, val_loader, device)
        t2i_recall = val_t2i_mean_recall(val_recall)   # get text-to-image recall for early stopping

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_recall_list.append(val_recall)

        print(f"Epoch {epoch+1}/{args.epochs} - \
                train_loss: {train_loss:.4f} - \
                val_loss: {val_loss:.4f} - \
                val_t2i_mean_recall: {t2i_recall:.4f} - \
                val_recall@k: {val_recall}")

        if t2i_recall > best_t2i_recall:
            best_t2i_recall = t2i_recall
            best_recall = val_recall
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Chosen model - val_t2i_mean_recall: {best_t2i_recall:.4f} - val_recall@k: {best_recall}")
    return train_loss_list, val_loss_list, val_recall_list

def train_one_epoch(device, model, optimizer, train_loader):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        text_embeds = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        image_embeds = model.get_image_features(pixel_values=pixel_values)

        logit_scale = model.logit_scale.exp()

        sample_weights = batch.get("sample_weights")
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)

        loss = contrastive_loss(image_embeds, text_embeds, logit_scale, sample_weights)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

if __name__ == '__main__':
    pass