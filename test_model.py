import torch
from datasets import load_dataset
from transformers import AutoProcessor

from data_loaders.mami_dataset import MAMIDataset
from models.clip_baseline import CLIPBaseline


MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

print("Loading MAMI...")
hf_dataset = load_dataset("scintist/MAMI-dataset")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

dataset = MAMIDataset(
    hf_dataset["train"],
    processor
)

sample = dataset[0]

input_ids = sample["input_ids"].unsqueeze(0).to(device)
attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
pixel_values = sample["pixel_values"].unsqueeze(0).to(device)

print("Loading CLIP model...")

model = CLIPBaseline().to(device)
model.eval()

with torch.no_grad():

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )

print("Logits shape:", logits.shape)
print("Logits:", logits)

prediction = torch.argmax(logits, dim=1)

print("Prediction:", prediction.item())
print("Actual label:", sample["label"].item())