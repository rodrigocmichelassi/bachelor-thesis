import torch
import torch.nn.functional as F
import numpy as np
from src.training.loss import contrastive_loss, retrieval_loss_ids
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

def retrieval_metrics(i2t_top_k, t2i_top_k, relevance, ks):
    """Score fixed rankings against an image-by-caption relevance matrix.

    Recall@K: Out of the K images, is any of them relevant?
    Precision@K: What percentage of the returned images are relevant
    Set Recall@K: What percentage of the relevant images was returned?

    If K exceeds the candidate count, precision uses the number actually
    returned, min(K, number of candidates).
    """
    results = {metric: {"i2t": {}, "t2i": {}}
               for metric in ("recall", "precision", "set_recall")}
    for direction, top_k, relevant in (("i2t", i2t_top_k, relevance),
                                      ("t2i", t2i_top_k, relevance.T)):
        hits = relevant.gather(1, top_k)
        relevant_count = relevant.sum(dim=1)
        for k in ks:
            found = hits[:, :k].sum(dim=1).float()
            returned = min(k, relevant.shape[1])

            results["recall"][direction][k] = (found > 0).float().mean().item()
            results["precision"][direction][k] = (found / returned).mean().item()
            results["set_recall"][direction][k] = (found / relevant_count).mean().item()
    return results

def add_identical_caption_relevance(matches, caption_to_img, token_sequences):
    """Add direct matches through identical caption tokens across the full split.

    Batch-local text IDs cannot be compared across batches. Instead, token tuples
    identify equal captions globally. No transitive merging of groups is done.
    """
    by_text = {}
    for caption_index, tokens in enumerate(token_sequences):
        by_text.setdefault(tokens, []).append(caption_index)
    result = matches.clone()
    for columns in by_text.values():
        if len(columns) > 1:
            columns = torch.tensor(columns, device=matches.device)
            images = caption_to_img[columns].unique()
            result[images[:, None], columns[None, :]] = True
    return result


# Embeddings and rankings cover the complete evaluation split, not each batch.
def evaluate_retrieval(model, test_loader, device, ks=(1, 5, 10), debug_oracle=False,
                       mask_same_group=False, retrieval_match_mode="exact"):
    model.to(device)
    model.eval()
    img_embs, txt_embs, ids = [], [], []
    group_ids, token_sequences = [], []
    if retrieval_match_mode not in ("exact", "group_text"):
        raise ValueError(f"Unknown retrieval match mode: {retrieval_match_mode}")

    running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            image_embeds = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
            text_embeds = model.get_text_features(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            running_loss += contrastive_loss(
                image_embeds, text_embeds, model.logit_scale.exp(),
                **retrieval_loss_ids(batch, device, mask_same_group)
            ).item()

            img_embs.append(F.normalize(image_embeds, dim=-1))
            txt_embs.append(F.normalize(text_embeds, dim=-1))

            ids.extend(batch["image_ids"])

            if retrieval_match_mode != "exact":
                if "group_ids" not in batch:
                    raise ValueError("Group-aware evaluation requires group_ids; use exact matching for datasets without groups.")
                group_ids.extend(batch["group_ids"].tolist())
            if retrieval_match_mode == "group_text":
                tokens = batch["input_ids"].detach().cpu()
                attention = batch["attention_mask"].detach().cpu().bool()
                token_sequences.extend(tuple(row[mask].tolist()) for row, mask in zip(tokens, attention))

    img_embs, txt_embs = torch.cat(img_embs), torch.cat(txt_embs)
    if debug_oracle:
        img_embs = txt_embs.clone()

    # Keep one embedding per image, including modes with several captions/image.
    first_row = {}
    for i, img_id in enumerate(ids):
        first_row.setdefault(img_id, i)

    unique_ids = list(first_row)
    first_indices = list(first_row.values())
    unique_img_embs = img_embs[first_indices]
    img_index = {img_id: n for n, img_id in enumerate(unique_ids)}
    caption_to_img = torch.tensor([img_index[i] for i in ids], device=device)

    similarity = unique_img_embs @ txt_embs.T

    i2t_top_k = similarity.topk(min(max(ks), similarity.shape[1]), dim=1).indices
    t2i_top_k = similarity.T.topk(min(max(ks), similarity.shape[0]), dim=1).indices

    image_indices = torch.arange(len(unique_ids), device=device)

    # boolean matrix, to which image each caption belongs to
    # rows: images; columns: captions
    exact_matches = image_indices[:, None] == caption_to_img[None, :]
    results = {
        "loss": running_loss / len(test_loader),
        "exact": retrieval_metrics(i2t_top_k, t2i_top_k, exact_matches, ks),
    }

    if retrieval_match_mode != "exact":
        caption_groups = torch.tensor(group_ids, device=device)
        image_groups = caption_groups[first_indices]
        group_matches = exact_matches | (image_groups[:, None] == caption_groups[None, :])

        results["group"] = retrieval_metrics(i2t_top_k, t2i_top_k, group_matches, ks)
        if retrieval_match_mode == "group_text":
            group_text_matches = add_identical_caption_relevance(group_matches, caption_to_img, token_sequences)
            results["group_text"] = retrieval_metrics(i2t_top_k, t2i_top_k, group_text_matches, ks)

    return results

# mean_recall returns the mean value between Recall@K,
# for all K = {1, 5, 10}
def mean_recall(results):
    values = [value for direction in ("i2t", "t2i") for value in results["recall"][direction].values()]
    return sum(values) / len(values)

# show_retrieval_metrics prints the metrics for both text-to-image
# retrieval and image-to-text retrieval
def show_retrieval_metrics(results, ks=(1, 5, 10)):
    print("\n-*-*- Test results -*-*-\n")
    print(f"Testing loss: {results['loss']:.4f}")

    match_mode_dict = {
        "exact": "exact match",
        "group_text": "group match"
    }

    for match_mode in ("exact", "group_text"):
        if match_mode not in results:
            continue

        metrics = results[match_mode]
        print(f"\nMatch mode: {match_mode_dict[match_mode]}")

        for direction, label in (("t2i", "Text -> Image"), ("i2t", "Image -> Text")):
            print(f"\n{label}")
            print(f"{'K':>4} {'Recall@K':>12} {'Precision@K':>14} {'Set Recall@K':>15}")

            for k in ks:
                recall = metrics["recall"][direction][k]
                precision = metrics["precision"][direction][k]
                set_recall = metrics["set_recall"][direction][k]
                print(f"{k:>4} {recall:>12.2%} {precision:>14.2%} {set_recall:>15.2%}")

            direction_mean = np.mean(list(metrics["recall"][direction].values()))
            print(f"Mean Recall@K: {direction_mean:.2%}")

        print(f"\nMean Recall@K (both directions): {mean_recall(metrics):.2%}")

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
