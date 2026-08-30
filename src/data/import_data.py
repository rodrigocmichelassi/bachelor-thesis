import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from functools import partial

from src.config import CLASSIFICATION_CAPTIONS_CSV, IMAGES_DIR
from src.dataset import RetinalClassCaptionDataset

# collate_fn turns the list of samples into pre-processed tensor batches
def collate_fn(batch, processor):
    images, texts, class_labels = zip(*batch)

    # Wraps an image feature extractor and a text tokeziner
    # Normalize and resizes images
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",    # pytorch tensors
        padding=True,
    )

    inputs["class_labels"] = list(class_labels)

    # Batches of (image/text) tensors and attention_mask (for text and image, due to padding)
    # wraps `pixel_values`, `input_ids`, `attention_mask` and `class_labels`.
    return inputs

# get_dataloaders get a list of (image, text) pair dataloaders for model training
def get_dataloaders(train_dataset, test_dataset, val_dataset):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor),
    )

    return train_loader, test_loader, val_loader

# get_df_split defines train, validation and test set splits
def get_df_split(df, debug):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["class"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["class"],
        random_state=42,
    )

    if debug:
        print(len(train_df), len(val_df), len(test_df))

    return train_df, test_df, val_df

# import_data is responsible for importing and pre-processing 
# the data used to train the models
def import_data(debug=False):
    captions_df = pd.read_csv(CLASSIFICATION_CAPTIONS_CSV)

    train_df, test_df, val_df = get_df_split(captions_df, debug)

    train_dataset = RetinalClassCaptionDataset(train_df, IMAGES_DIR)
    val_dataset = RetinalClassCaptionDataset(val_df, IMAGES_DIR)
    test_dataset = RetinalClassCaptionDataset(test_df, IMAGES_DIR)

    train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset)

    batch = next(iter(train_loader))

    if debug:
        print(batch.keys())
        print(batch["pixel_values"].shape)
        print(batch["input_ids"].shape)

    return train_loader, test_loader, val_loader
        
if __name__ == "__main__":
    _, _, _ = import_data(debug=True)