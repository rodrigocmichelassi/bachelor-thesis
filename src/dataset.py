import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SingleRowRetrievalDataset(Dataset):
    def __init__(self, captions, image_dir):
        self.image_dir = image_dir
        self.captions = captions.reset_index(drop=True)
        self.caption_cols = ["disease_caption", "anatomy_caption", "quality_caption"]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]

        image_path = f'{self.image_dir}/{row["image_id"]}'
        image = Image.open(image_path).convert("RGB")

        caption = row[random.choice(self.caption_cols)]

        return image, caption, row["image_id"]


class MultiRowRetrievalDataset(Dataset):
    def __init__(self, captions, image_dir):
        self.image_dir = image_dir
        self.captions = captions.reset_index(drop=True)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]

        image_path = f"{self.image_dir}/{row['image_id']}"
        image = Image.open(image_path).convert("RGB")

        caption = row["caption"]

        return image, caption, row["image_id"]

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