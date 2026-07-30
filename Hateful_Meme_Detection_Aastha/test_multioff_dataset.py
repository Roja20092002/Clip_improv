from datasets.multioff_dataset import MultiOFFDataset
from utils.image_transforms import get_default_image_transform

dataset = MultiOFFDataset(
    data_root="data/MultiOFF",
    split="train",
    transform=get_default_image_transform()
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nKeys:", sample.keys())
print("Image shape:", sample["image"].shape)
print("Text:", sample["text"])
print("Label:", sample["labels"])
print("Task ID:", sample["task_ids"])
