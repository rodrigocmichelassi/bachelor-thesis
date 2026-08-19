import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from functools import partial

from src.config import CLASSIFICATION_CAPTIONS_CSV, IMAGES_DIR
from src.dataset import RetinalClassCaptionDataset

def collate_fn(batch, processor):
    images, texts = zip(*batch)
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    return inputs

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

def import_data(debug=False):
    df = pd.read_csv(CLASSIFICATION_CAPTIONS_CSV)

    train_df, test_df, val_df = get_df_split(df, debug)

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