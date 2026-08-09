import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor


class HatefulMemesDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        image_dir,
        model_name="openai/clip-vit-base-patch32",
    ):
        self.data = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")

        text = row["text"]
        label = int(row["label"])

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }