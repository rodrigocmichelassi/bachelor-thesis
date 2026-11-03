import torch
import torch.nn.functional as F

def contrastive_loss(image_embeds, text_embeds, logit_scale, sample_weights=None,
                     *, group_ids=None, text_ids=None):
    """Symmetric InfoNCE; optionally ignore equivalent off-diagonal pairs.

    Equivalent pairs are excluded as negatives, not added as positives.
    Omitting both ID tensors preserves the original objective.
    """
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    
    # mask same group_ids in the batch
    if group_ids is not None or text_ids is not None:
        same = torch.zeros_like(logits_per_image, dtype=torch.bool)
        for ids in (group_ids, text_ids):
            if ids is not None:
                if ids.ndim != 1 or ids.shape[0] != logits_per_image.shape[0]:
                    raise ValueError("Loss IDs must have one entry per image-caption pair.")
                ids = ids.to(logits_per_image.device)
                same |= ids[:, None] == ids[None, :]
        same.fill_diagonal_(False)
        logits_per_image = logits_per_image.masked_fill(same, float("-inf"))

    logits_per_text = logits_per_image.T

    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)

    loss_i = F.cross_entropy(logits_per_image, labels, reduction="none")  # [batch_size] - one loss value per sample
    loss_t = F.cross_entropy(logits_per_text, labels, reduction="none")   # [batch_size] - one loss value per sample

    per_sample_loss = (loss_i + loss_t) / 2

    if sample_weights is not None:
        per_sample_loss = per_sample_loss * sample_weights  # give more weights to rare samples
        return per_sample_loss.sum() / sample_weights.sum()
    return per_sample_loss.mean()


def retrieval_loss_ids(batch, device, enabled=False):
    """Build the optional loss arguments identically for train and evaluation."""
    if not enabled:
        return {}
    if "group_ids" not in batch:
        raise ValueError("Group masking requires group_ids in every retrieval batch.")
    
    # Compare the actual token sequences seen by CLIP, ignoring padding values.
    # text_ids -> give the same ID to equal sentences.
    tokens = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -1)
    _, text_ids = torch.unique(tokens, dim=0, return_inverse=True)

    return {"group_ids": batch["group_ids"].to(device), "text_ids": text_ids.to(device)}
