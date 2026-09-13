import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.training.loss import contrastive_loss
from src.training.evaluate import evaluate_dataset
from src.config import CLIP_BEST_MODEL

def contrastive_loss(image_embeds, text_embeds, logit_scale):
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    logits_per_text = logits_per_image.T

    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2

def train_classifier(args, device, model, train_loader, val_loader, processor, class_labels):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    max_acc = 0.0
    chosen_loss = 0.0

    patience = 7
    epochs_without_improvement = 0

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(args.epochs):
        contrastive_train_loss = train_one_epoch(device, model, optimizer, train_loader)

        model.eval()
        train_true_labels, train_pred_labels, train_loss = evaluate_dataset(model, processor, train_loader, class_labels, device)
        val_true_labels, val_pred_labels, val_loss = evaluate_dataset(model, processor, val_loader, class_labels, device)

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
            model.save_pretrained(CLIP_BEST_MODEL)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                print(f"Early stopping at epoch: {epoch}")
                break

    print(f"Chosen model - val_acc: {max_acc} - val_loss: {chosen_loss}")

    return train_acc_list, train_loss_list, val_acc_list, val_loss_list

def train_one_epoch(device, model, optimizer, train_loader):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        text_embeds = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        image_embeds = model.get_image_features(pixel_values=pixel_values)

        logit_scale = model.logit_scale.exp()

        loss = contrastive_loss(image_embeds, text_embeds, logit_scale)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

if __name__ == '__main__':
    pass