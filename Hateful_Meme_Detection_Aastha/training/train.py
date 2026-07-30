import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.mtl_model import MTLModel
from datasets.loaders import get_hateful_memes_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    # =====================================================
    # 1. Device
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =====================================================
    # 2. Model
    # =====================================================
    model = MTLModel(num_tasks=4)
    model.to(device)
    model.train()
    print("Model initialized")

    # =====================================================
    # 3. Loss & Optimizer
    # =====================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    print("Loss & optimizer ready")

    # =====================================================
    # 4. Resume from checkpoint (optional)
    # =====================================================
    checkpoint_path = "checkpoints/mtl_hateful_memes.pt"
    start_step = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint.get("step", 0)
        print(f"Resumed from checkpoint at step {start_step}")
    else:
        print("No checkpoint found, training from scratch")

    # =====================================================
    # 5. Dataloader (TRAIN)
    # =====================================================
    train_loader = get_hateful_memes_loader(
        data_root="data/HatefulMemes/data",
        split="train",
        batch_size=2,
        transform=get_default_image_transform(),
    )
    print("Training dataloader ready")

    # =====================================================
    # 6. Tokenizer
    # =====================================================
    tokenizer = get_bert_tokenizer()

    # =====================================================
    # 7. Training Loop
    # =====================================================
    num_epochs = 1
    global_step = start_step

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch in train_loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)      # Tensor
            task_ids = batch["task_ids"]              # Tensor

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
                "pixel_values": images,
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
                print(f"Step {global_step} | Loss: {loss.item():.4f}")

            global_step += 1

    # =====================================================
    # 8. Validation (DEV)
    # =====================================================
    model.eval()
    print("\nRunning validation...")

    val_loader = get_hateful_memes_loader(
        data_root="data/HatefulMemes/data",
        split="dev",
        batch_size=4,
        transform=get_default_image_transform(),
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)     # Tensor
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
                "pixel_values": images,
            }

            logits = model(
                clip_inputs=clip_inputs,
                uniter_inputs=clip_inputs,
                bert_inputs=tokenized,
                task_id=task_ids[0].item()
            )

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy (Hateful Memes): {val_acc:.4f}")

    # =====================================================
    # 9. Save checkpoint
    # =====================================================
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": global_step,
        "val_accuracy": val_acc
    }

    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved successfully")


if __name__ == "__main__":
    main()
