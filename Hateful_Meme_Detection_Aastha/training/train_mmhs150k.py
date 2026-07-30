import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.mtl_model import MTLModel
from datasets.loaders import get_mmhs150k_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    # --------------------------------------------------
    # 1. Device
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------
    # 2. Model
    # --------------------------------------------------
    model = MTLModel(num_tasks=8)  # IMPORTANT: must include task_id = 7
    model.to(device)
    model.train()

    print("Model initialized")

    # --------------------------------------------------
    # 3. Loss + Optimizer
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    print("Loss & optimizer ready")

    # --------------------------------------------------
    # 4. DataLoader (MMHS150K)
    # --------------------------------------------------
    train_loader = get_mmhs150k_loader(
        data_root="data/MMHS150K",
        split="train",
        batch_size=8,
        transform=get_default_image_transform(),
    )

    print("MMHS150K train loader ready")

    # --------------------------------------------------
    # 5. Tokenizer
    # --------------------------------------------------
    tokenizer = get_bert_tokenizer()

    # --------------------------------------------------
    # 6. Training loop
    # --------------------------------------------------
    num_epochs = 1
    global_step = 0
    TASK_ID = 7  # MMHS150K

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch in tqdm(train_loader):
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)

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
                task_id=TASK_ID
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if global_step % 50 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")

            global_step += 1

    # --------------------------------------------------
    # 7. Save checkpoint
    # --------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        "checkpoints/mtl_mmhs150k.pt"
    )

    print("MMHS150K training completed & checkpoint saved")


if __name__ == "__main__":
    main()
