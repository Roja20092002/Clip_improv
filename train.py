import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.clip_baseline import CLIPBaseline
from datasets.hateful_memes_dataset import HatefulMemesDataset
from utils.metrics import calculate_metrics
from utils.seed import set_seed
import configs.config as config


set_seed(config.SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CLIPBaseline().to(device)

train_dataset = HatefulMemesDataset(
    annotations_file="data/train.csv",
    image_dir="data/images",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
)

for epoch in range(config.EPOCHS):

    model.train()

    running_loss = 0

    predictions = []
    labels = []

    progress = tqdm(train_loader)

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

        predictions.extend(pred.cpu().numpy())
        labels.extend(target.cpu().numpy())

        progress.set_description(
            f"Epoch {epoch+1}/{config.EPOCHS}"
        )

    metrics = calculate_metrics(labels, predictions)

    print(
        f"Loss: {running_loss:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )

torch.save(
    model.state_dict(),
    "checkpoints/clip_baseline.pth",
)