import torch
import torch.nn.functional as F
import numpy as np
# from tqdm import tqdm
from src.training.loss import contrastive_loss, retrieval_loss_ids
from src.training.evaluate import evaluate_classifier, evaluate_retrieval
from src.utils.helper import get_run_checkpoint_path, get_retrieval_match_mode

# val_t2i_mean_recall returns the mean text-to-image Recall@K value
# for a given match mode.
def val_t2i_mean_recall(results, match_mode='exact'):
    selected = results[match_mode]
    t2i = selected["recall"]["t2i"]

    return sum(t2i.values()) / len(t2i)

# print_retrieval_epoch_metrics prints the current epoch and validation
# metrics obtained. It also marks with "| best"
def print_retrieval_epoch_metrics(epoch, epochs, train_loss, results, match_mode, is_best, ks=(1, 5, 10)):
    match_mode_dict = {
        "exact": "exact match",
        "group_text": "group match"
    }

    metrics = results[match_mode]
    k_label = '/'.join(str(k) for k in ks)
    fields = [
        f"Epoch {epoch}/{epochs}",
        f"train_loss: {train_loss:.4f}",
        f"val_loss: {results['loss']:.4f}",
        f"{match_mode_dict[match_mode]} text->image",
    ]

    for key, label in (("recall", "Recall"), ("precision", "Precision"), ("set_recall", "SetRecall")):
        values = '/'.join(f"{metrics[key]['t2i'][k]:.2%}" for k in ks)
        fields.append(f"{label}@{k_label}: {values}")
    
    if is_best:
        fields.append("best")
    print(' | '.join(fields))

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
    best_epoch = 0
    patience = 7
    epochs_without_improvement = 0
    checkpoint_path = get_run_checkpoint_path(args)

    train_loss_list, val_loss_list, val_metrics_list = [], [], []
    mask_same_group = args.mask_same_group
    match_mode = get_retrieval_match_mode(args)
    print(f"Checkpoint selection: {match_mode} text-to-image mean Recall@1/5/10")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(device, model, optimizer, train_loader,
                                     mask_same_group=mask_same_group)

        val_results = evaluate_retrieval(model, val_loader, device,
                                                 mask_same_group=mask_same_group,
                                                 retrieval_match_mode=match_mode)
        val_loss = val_results["loss"]
        t2i_recall = val_t2i_mean_recall(val_results, match_mode)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_metrics_list.append(val_results[match_mode])

        is_best = t2i_recall > best_t2i_recall
        print_retrieval_epoch_metrics(epoch + 1, args.epochs, train_loss, val_results, match_mode, is_best)

        if is_best:
            best_t2i_recall = t2i_recall
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Chosen model: epoch {best_epoch} | {match_mode} text->image | "
          f"mean Recall@1/5/10: {best_t2i_recall:.2%}")

    return train_loss_list, val_loss_list, val_metrics_list

def train_one_epoch(device, model, optimizer, train_loader, mask_same_group=False):
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

        loss = contrastive_loss(image_embeds, text_embeds, logit_scale, sample_weights,
                                **retrieval_loss_ids(batch, device, mask_same_group))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

if __name__ == '__main__':
    pass
