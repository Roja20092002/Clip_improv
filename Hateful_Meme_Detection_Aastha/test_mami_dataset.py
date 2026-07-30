import torch
from datasets.mami_dataset import MAMIDataset
from utils.image_transforms import get_default_image_transform

# Path to MAMI data
DATA_ROOT = "data/MAMI/data"

# Create dataset
dataset = MAMIDataset(
    data_root=DATA_ROOT,
    split="train",
    transform=get_default_image_transform()
)

print("Dataset length:", len(dataset))

# Get one sample
sample = dataset[0]

print("\nKeys in one sample:")
for k in sample.keys():
    print("-", k)

print("\nImage shape:", sample["image"].shape)
print("Text:", sample["text"])
print("Labels:", sample["labels"])
print("Task IDs:", sample["task_ids"])

