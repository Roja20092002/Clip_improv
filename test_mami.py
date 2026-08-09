from datasets import load_dataset
from transformers import AutoProcessor

from data_loaders.mami_dataset import MAMIDataset


MODEL_NAME = "openai/clip-vit-base-patch32"


print("Loading MAMI...")

hf_dataset = load_dataset("scintist/MAMI-dataset")

print(hf_dataset)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

dataset = MAMIDataset(
    hf_dataset["train"],
    processor
)

print("MAMI size:", len(dataset))

sample = dataset[0]

print("\nSample keys:")
print(sample.keys())

print("\nInput IDs shape:")
print(sample["input_ids"].shape)

print("\nAttention mask shape:")
print(sample["attention_mask"].shape)

print("\nPixel values shape:")
print(sample["pixel_values"].shape)

print("\nLabel:")
print(sample["label"])