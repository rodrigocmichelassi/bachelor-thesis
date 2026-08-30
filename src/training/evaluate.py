import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# evaluate_dataset calculates the cosine similarity between
# each batch image embeddings and all the possible labels embeddings
# and gets the closest mapping from each image to each text to perform
# a classification using CLIP.
def evaluate_dataset(model, processor, test_loader, class_labels, device):
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

    text_sim = text_embeds @ text_embeds.T  # [13, 13]
    print(text_sim) 

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in test_loader:

            pixel_values = batch["pixel_values"].to(device)
            batch_true_labels = batch["class_labels"]

            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)  # image_embeds: [32, embed_dim]

            logit_scale = model.logit_scale.exp()
            # calculates clip similarity matrix
            similarity = logit_scale * image_embeds @ text_embeds.T  # cosine similarity, [32, 13], CLIP matrix

            # for each row (batch image), gets the max score for cosine similarity
            # (classification label for each image)
            batch_predicted_idx = similarity.argmax(dim=1)
            batch_predicted_labels = [class_names[i] for i in batch_predicted_idx]

            true_labels.extend(batch_true_labels)
            predicted_labels.extend(batch_predicted_labels)

    return true_labels, predicted_labels

def calculate_classification_metrics(true_labels, pred_labels, class_labels, print_results=True):
    class_names = list(class_labels.keys())

    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=class_names, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    if print_results:
        print(f'Accuracy: {accuracy}')
        print(f'Balanced Accuracy: {balanced_accuracy}')
        print(f'Classification Report: \n{report}')
        print(f'Confusion Matrix: \n{cm}')

    return accuracy, balanced_accuracy, report, cm

if __name__ == '__main__':
    pass