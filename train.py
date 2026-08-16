import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

from models.clip_baseline import CLIPBaseline
from data_loaders.mami_dataset import MAMIDataset
from utils.metrics import calculate_metrics
from utils.seed import set_seed
import configs.config as config


set_seed(config.SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# -------------------------
# Load MAMI
# -------------------------

print("Loading MAMI dataset...")

hf_dataset = load_dataset("scintist/MAMI-dataset")

processor = AutoProcessor.from_pretrained(
    config.MODEL_NAME,
    use_fast=True
)

train_dataset = MAMIDataset(
    hf_dataset["train"],
    processor
)

print("Training samples:", len(train_dataset))


# -------------------------
# DataLoader
# -------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# -------------------------
# Model
# -------------------------

print("Loading CLIP model...")

model = CLIPBaseline(
    num_classes=config.NUM_CLASSES
).to(device)


# -------------------------
# Loss + optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

for epoch in range(config.EPOCHS):

    model.train()

    running_loss = 0.0

    predictions = []
    labels = []

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{config.EPOCHS}"
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        target = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        loss = criterion(logits, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pred = torch.argmax(logits, dim=1)

        predictions.extend(
            pred.detach().cpu().numpy()
        )

        labels.extend(
            target.detach().cpu().numpy()
        )

        progress.set_postfix(
            loss=loss.item()
        )

    metrics = calculate_metrics(
        labels,
        predictions
    )

    average_loss = running_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1}"
        f" | Loss: {average_loss:.4f}"
        f" | Accuracy: {metrics['accuracy']:.4f}"
        f" | Precision: {metrics['precision']:.4f}"
        f" | Recall: {metrics['recall']:.4f}"
        f" | F1: {metrics['f1']:.4f}"
    )


# -------------------------
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "checkpoints/clip_baseline_mami.pth"
)

print("\nModel saved.")