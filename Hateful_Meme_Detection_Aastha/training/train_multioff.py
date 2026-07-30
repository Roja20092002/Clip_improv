import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.mtl_model import MTLModel
from datasets.loaders import get_multioff_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MTLModel(num_tasks=7)  # MultiOFF task_id = 6
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    loader = get_multioff_loader(
        data_root="data/MultiOFF",
        split="train",
        batch_size=4,
        transform=get_default_image_transform()
    )

    tokenizer = get_bert_tokenizer()

    num_epochs = 1
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch in loader:
            images = batch["image"].to(device)
            texts = batch["text"]
            labels = batch["labels"].to(device)
            task_ids = batch["task_ids"]

            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            clip_inputs = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "pixel_values": images
            }

            optimizer.zero_grad()

            logits = model(
                clip_inputs=clip_inputs,
                uniter_inputs=clip_inputs,
                bert_inputs=tokenized,
                task_id=task_ids[0].item()
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                print(f"Step {global_step} | Total Loss: {loss.item():.4f}")

            global_step += 1

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mtl_multioff.pt")
    print("MultiOFF training completed & checkpoint saved")


if __name__ == "__main__":
    main()
