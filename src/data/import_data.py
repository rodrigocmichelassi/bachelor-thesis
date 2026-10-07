import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from functools import partial

from src.config import CLASSIFICATION_CAPTIONS_CSV, IMAGES_DIR
from src.dataset import RetinalClassCaptionDataset

# seed_worker sets a seed for pytorch dataloader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# collate_fn turns the list of samples into pre-processed tensor batches
def collate_fn(batch, processor, class_weights_dict=None):
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
    if class_weights_dict is not None:
        inputs["sample_weights"] = torch.tensor([class_weights_dict[c] for c in class_labels], dtype=torch.float)
 
    # Batches of (image/text) tensors and attention_mask (for text and image, due to padding)
    # wraps `pixel_values`, `input_ids`, `attention_mask` and `class_labels`.
    return inputs

# get_dataloaders get a list of (image, text) pair dataloaders for model training
def get_dataloaders(train_dataset, test_dataset, val_dataset, class_weights_dict, seed):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor, class_weights_dict=class_weights_dict),
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=16,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=16,
        pin_memory=True,
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
def import_data(args, debug=False):
    captions_df = pd.read_csv(CLASSIFICATION_CAPTIONS_CSV)

    train_df, test_df, val_df = get_df_split(captions_df, debug)

    # class weights: 1 / count or 1 / sqrt(count) per class
    # to help weight the least frequent classes, without overvaluing these.
    class_counts = train_df["class"].value_counts()

    class_weights_dict = None
    if args.weight_classes == 1:
        class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
    elif args.weight_classes == 2:
        class_weights_dict = {cls: 1.0 / (count**0.5) for cls, count in class_counts.items()} 

    train_dataset = RetinalClassCaptionDataset(train_df, IMAGES_DIR)
    val_dataset = RetinalClassCaptionDataset(val_df, IMAGES_DIR)
    test_dataset = RetinalClassCaptionDataset(test_df, IMAGES_DIR)

    train_loader, test_loader, val_loader = get_dataloaders(
        train_dataset, test_dataset, val_dataset, class_weights_dict, args.seed)


    if debug:
        batch = next(iter(train_loader))
        
        print(batch.keys())
        print(batch["pixel_values"].shape)
        print(batch["input_ids"].shape)

    return train_loader, test_loader, val_loader
        
if __name__ == "__main__":
    _, _, _ = import_data(debug=True)