import torch
import torch.nn.functional as F

def contrastive_loss(image_embeds, text_embeds, logit_scale, sample_weights=None):
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_embeds @ text_embeds.T
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