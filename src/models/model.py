from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, PeftModel

def load_raw_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor

def load_lora_clip(debug=False):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    config = LoraConfig(
        r=8,
        lora_alpha=16, # ΔW = (alpha / r) * A @ B -> usually alpha = 2*r
        target_modules=["q_proj", "v_proj"],  # attention projections, as per the LoRA paper
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, config)
    
    if debug:
        model.print_trainable_parameters()
    
    return model, processor

def load_fine_tuned_clip(adapter_path, debug=False):
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    if debug:
        model.print_trainable_parameters()

    return model, processor

if __name__ == '__main__':
    pass