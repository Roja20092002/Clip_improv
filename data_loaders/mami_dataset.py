import torch
from torch.utils.data import Dataset


class MAMIDataset(Dataset):

    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        image = sample["image"]
        text = sample["text"]

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        # MAMI binary classification:
        # 0 -> non-misogynous
        # >0 -> misogynous

        binary_label = (
            0
            if sample["multiclass_label"] == 0
            else 1
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "pixel_values":
                encoding["pixel_values"].squeeze(0),

            "label":
                torch.tensor(
                    binary_label,
                    dtype=torch.long
                ),
        }