from datasets.mmhs150k_dataset import MMHS150KDataset
from utils.image_transforms import get_default_image_transform

dataset = MMHS150KDataset(
    data_root="data/MMHS150K",
    split="train",
    transform=get_default_image_transform()
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nKeys:", sample.keys())
print("Image shape:", sample["image"].shape)
print("Text:", sample["text"][:100])
print("Label:", sample["labels"])
print("Task ID:", sample["task_ids"])
