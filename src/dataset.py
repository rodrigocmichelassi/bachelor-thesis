import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageTextDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.captions[idx]
        return image, text

class RetinalClassCaptionDataset(Dataset):
    def __init__(self, captions, image_dir):
        self.captions = captions.reset_index(drop=True)
        self.image_dir = image_dir
        self.caption_cols = ["caption_1", "caption_2", "caption_3"]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]

        image_path = f"{self.image_dir}/{row['image_id']}"
        image = Image.open(image_path).convert("RGB")

        # sample a different caption each time this item is fetched
        caption = row[random.choice(self.caption_cols)]
        class_label = row["class"]

        return image, caption, class_label