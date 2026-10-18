import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# evaluate_classifier calculates the cosine similarity between
# each batch image embeddings and all the possible labels embeddings
# and gets the closest mapping from each image to each text to perform
# a classification using CLIP.
def evaluate_classifier(model, processor, test_loader, class_labels, device, debug=False):
    # Batch size: 32, len(class_labels): 13
    model.to(device)
    model.eval()

    class_names = list(class_labels.keys())
    prompts = list(class_labels.values())

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    # Get text embeddings from pre-defined text labels
    # normalize to calculate cosine similarity
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) # text_embeds: [13, embed_dim]

    if debug:
        text_similarity = text_embeds @ text_embeds.T  # [13, 13]
        print(text_similarity) 

    true_labels = []
    predicted_labels = []

    running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:

            pixel_values = batch["pixel_values"].to(device)
            batch_true_labels = batch["class_labels"]

            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)  # image_embeds: [batch_size, embed_dim]

            logit_scale = model.logit_scale.exp()
            # calculates clip similarity matrix
            similarity = logit_scale * image_embeds @ text_embeds.T  # cosine similarity, [batch_size, 13], CLIP matrix

            # classification loss: cross-entropy against the true class index
            true_idx = torch.tensor(
                [class_names.index(label) for label in batch_true_labels],
                device=device
            )
            loss = F.cross_entropy(similarity, true_idx)
            running_loss += loss.item()

            # for each row (batch image), gets the max score for cosine similarity
            # (classification label for each image)
            batch_predicted_idx = similarity.argmax(dim=1)
            batch_predicted_labels = [class_names[i] for i in batch_predicted_idx]

            true_labels.extend(batch_true_labels)
            predicted_labels.extend(batch_predicted_labels)

    loss = running_loss / len(test_loader)

    return true_labels, predicted_labels, loss

# evaluate_retrieval evaluates CLIP's performance on image retrieval
def evaluate_retrieval(model, test_loader, device, ks=(1, 5, 10), debug_oracle=False):
    model.to(device)
    model.eval()

    img_embs, txt_embs, ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            image_embeds = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
            text_embeds = model.get_text_features(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            img_embs.append(F.normalize(image_embeds, dim=-1))
            txt_embs.append(F.normalize(text_embeds, dim=-1))
            ids.extend(batch["image_ids"])

    img_embs = torch.cat(img_embs)   # one row per (image, caption) pair
    txt_embs = torch.cat(txt_embs)

    if debug_oracle:
        img_embs = txt_embs.clone()

    # Keep the first embedding per unique image (each image appears 3x)
    first_row = {}
    for i, img_id in enumerate(ids):
        first_row.setdefault(img_id, i)
    unique_ids = list(first_row.keys())
    unique_img_embs = img_embs[[first_row[i] for i in unique_ids]]

    img_index = {img_id: n for n, img_id in enumerate(unique_ids)}
    caption_to_img = torch.tensor([img_index[i] for i in ids], device=device)  # [N_captions]

    similarity = unique_img_embs @ txt_embs.T   # [N_images, N_captions]
    max_k = max(ks)

    # image -> text: hit if ANY of the image's captions is in the top K
    i2t_top = similarity.topk(max_k, dim=1).indices # retrieve the top K captions
    i2t_hits = caption_to_img[i2t_top] == torch.arange(similarity.shape[0], device=device).unsqueeze(1) # [N_images, max_k] boolean matrix

    # text -> image: hit if the caption's image is in the top K
    t2i_top = similarity.T.topk(max_k, dim=1).indices # retrieve the top K images
    t2i_hits = t2i_top == caption_to_img.unsqueeze(1)

    # image -> text hits if any of the captions is in the top K
    # text -> image hits if the corresponding image to the caption is in the top K (harder)
    return {
        "i2t": {k: i2t_hits[:, :k].any(dim=1).float().mean().item() for k in ks},   # dict with Recall@K -> {1: 0.12, 5: 0.35, 10: 0.48}
        "t2i": {k: t2i_hits[:, :k].any(dim=1).float().mean().item() for k in ks},
    }

# mean_recall calculates the mean between i2t and t2i Recall@K results
def mean_recall(results):
    values = [v for direction in results.values() for v in direction.values()]
    return sum(values) / len(values)

# show_retrieval_metrics prints Recall@K in different formats
def show_retrieval_metrics(recall_at_K, ks=(1,5,10)):
    mean_recall_value = mean_recall(recall_at_K)
    i2t = recall_at_K['i2t']
    t2i = recall_at_K['t2i']

    i2t_mean = np.array(list(i2t.values())).mean()
    t2i_mean = np.array(list(t2i.values())).mean()
    
    print("\n-*-*- Test results -*-*-\n")
    print(f"Recall@K for K={ks}: {recall_at_K}")
    print(f"Image->Text Recall@K Mean: {i2t_mean}")
    print(f"Text->Image Recall@K Mean: {t2i_mean}")
    print(f"Mean Recall: {mean_recall_value}")

# calculate_classification_metrics calculates metrics for CLIP classification
def calculate_classification_metrics(true_labels, pred_labels, class_labels, print_results=True):
    class_names = list(class_labels.keys())

    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=class_names, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    if print_results:
        print("\n-*-*- Test results -*-*-\n")
        print(f'Accuracy: {accuracy}')
        print(f'Balanced Accuracy: {balanced_accuracy}')
        print(f'Classification Report: \n{report}')
        print(f'Confusion Matrix: \n{cm}')

    return accuracy, balanced_accuracy, report, cm

if __name__ == '__main__':
    pass