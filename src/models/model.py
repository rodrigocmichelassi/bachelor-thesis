from transformers import CLIPProcessor, CLIPModel

def load_raw_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor

if __name__ == '__main__':
    pass