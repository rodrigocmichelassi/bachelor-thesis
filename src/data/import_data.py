import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from functools import partial

from src.utils.helper import get_retrieval_dataset
from src.config import SR_RETRIEVAL_CAPTIONS_CSV, MR_RETRIEVAL_CAPTIONS_CSV, CLASSIFICATION_CAPTIONS_CSV, IMAGES_DIR
from src.dataset import RetinalClassCaptionDataset, SingleRowRetrievalDataset, MultiRowRetrievalDataset

# seed_worker sets a seed for pytorch dataloader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_captions_mode(args):
    if args.training_mode == 'classifier' or (args.training_mode == 'zero_shot' and args.zero_shot_mode == 'classifier'):
        return 'classifier'
    return 'retrieval'

# cls_collretrieval_collate_fnate_fn turns the list of samples into pre-processed tensor batches
def retrieval_collate_fn(batch, processor):
    images, texts, image_ids = zip(*batch)
    
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    )

    inputs["image_ids"] = list(image_ids)
    return inputs

# cls_collate_fn turns the list of samples into pre-processed tensor batches
def cls_collate_fn(batch, processor, class_weights_dict=None):
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
def get_dataloaders(args, train_dataset, test_dataset, val_dataset, class_weights_dict=None, seed=42):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    captions_to_load = get_captions_mode(args)

    if captions_to_load == 'classifier':
        # class weights apply to the train split only
        train_collate = partial(cls_collate_fn, processor=processor, class_weights_dict=class_weights_dict)
        eval_collate = partial(cls_collate_fn, processor=processor)
    
    elif captions_to_load == 'retrieval':
        train_collate = partial(retrieval_collate_fn, processor=processor)
        eval_collate = train_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=16,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=16,
        pin_memory=True,
    )

    return train_loader, test_loader, val_loader

# get_classification_df_split defines train, validation and test set splits
# for CLIP classification
def get_classification_df_split(df, seed, debug):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["class"],
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["class"],
        random_state=seed,
    )

    if debug:
        print(len(train_df), len(val_df), len(test_df))

    return train_df, test_df, val_df

# get_retrieval_df_split defines train, validation and test set splits
# for CLIP image retrieval
def get_retrieval_df_split(args, captions_df):
    # define train/test/val image_ids
    unique_images = captions_df["image_id"].unique()    
    train_ids, temp_ids = train_test_split(unique_images, test_size=0.3, random_state=args.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)

    # get all rows for the train/val/test image IDs
    train_df = captions_df[captions_df["image_id"].isin(train_ids)]
    val_df = captions_df[captions_df["image_id"].isin(val_ids)]
    test_df = captions_df[captions_df["image_id"].isin(test_ids)]

    return train_df, test_df, val_df

# import_classification_data returns data loaders for CLIP
# classification training, testing and validation
def import_classification_data(args, captions_df, debug=False):
    train_df, test_df, val_df = get_classification_df_split(captions_df, args.seed, debug)

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
        args, train_dataset, test_dataset, val_dataset, class_weights_dict, args.seed)


    if debug:
        batch = next(iter(train_loader))
        
        print(batch.keys())
        print(batch["pixel_values"].shape)
        print(batch["input_ids"].shape)

    return train_loader, test_loader, val_loader

# import_retrieval_data returns data loaders for CLIP
# image retrieval training, testing and validation
def import_retrieval_data(args, captions_df, debug=False):
    train_df, test_df, val_df = get_retrieval_df_split(args, captions_df)

    train_dataset, test_dataset, val_dataset = get_retrieval_dataset(args, train_df, test_df, val_df)

    train_loader, test_loader, val_loader = get_dataloaders(
            args, train_dataset, test_dataset, val_dataset, seed=args.seed)

    if debug:
        batch = next(iter(train_loader))
        
        print(batch.keys())
        print(batch["pixel_values"].shape)
        print(batch["input_ids"].shape)

    return train_loader, test_loader, val_loader

# import_data is responsible for importing and pre-processing 
# the data used to train the models
def import_data(args, debug=False):
    if get_captions_mode(args) == 'classifier':
        captions_df = pd.read_csv(CLASSIFICATION_CAPTIONS_CSV)
        return import_classification_data(args, captions_df, debug)

    elif get_captions_mode(args) == 'retrieval':
        if args.retrieval_mode == '':
            raise ValueError('Provide a retrieval mode in the parameters.')
        
        captions_df = pd.read_csv(SR_RETRIEVAL_CAPTIONS_CSV) \
            if args.retrieval_mode == 'single_row' else pd.read_csv(MR_RETRIEVAL_CAPTIONS_CSV)
        
        return import_retrieval_data(args, captions_df)
        
if __name__ == "__main__":
    pass