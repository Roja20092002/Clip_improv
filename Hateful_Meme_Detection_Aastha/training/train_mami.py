import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.mtl_model import MTLModel
from datasets.loaders import get_mami_loader
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
    model = MTLModel(num_tasks=6)  
    # task ids:
    # 1 = label
    # 2 = shaming
    # 3 = stereotype
    # 4 = objectification
    # 5 = violence

    model.to(device)
    model.train()
    print("Model initialized")

    # =====================================================
    # 3. Losses (ONE PER TASK)
    # =====================================================
    criterion = nn.CrossEntropyLoss()

    # =====================================================
    # 4. Optimizer
    # =====================================================
    optimizer = AdamW(model.parameters(), lr=2e-5)
    print("Optimizer ready")

    # =====================================================
    # 5. Load HM checkpoint (TRANSFER LEARNING)
    # =====================================================
    checkpoint_path = "checkpoints/mtl_hateful_memes.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        print("Loaded pretrained Hateful Memes weights")
    else:
        print("No HM checkpoint found — training from scratch")

    # =====================================================
    # 6. DataLoader (MAMI)
    # =====================================================
    train_loader = get_mami_loader(
        data_root="data/MAMI/data",
        split="train",
        batch_size=4,
        transform=get_default_image_transform(),
    )
    print("MAMI train loader ready")

    # =====================================================
    # 7. Tokenizer
    # =====================================================
    tokenizer = get_bert_tokenizer()

    # =====================================================
    # 8. Training Loop
    # =====================================================
    num_epochs = 1
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch in train_loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"]     # dict of tensors
            task_ids = batch["task_ids"] # dict of tensors

            # Move label tensors to GPU
            labels = {k: v.to(device) for k, v in labels.items()}

            # Tokenize text
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
            total_loss = 0.0

            # ================= MULTI-TASK LOSS =================
            for task_name in labels.keys():
                logits = model(
                    clip_inputs=clip_inputs,
                    uniter_inputs=clip_inputs,
                    bert_inputs=tokenized,
                    task_id=task_ids[task_name][0].item()
                )

                loss = criterion(logits, labels[task_name])
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                print(f"Step {global_step} | Total Loss: {total_loss.item():.4f}")

            global_step += 1

    # =====================================================
    # 9. Save MAMI checkpoint
    # =====================================================
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "steps": global_step
        },
        "checkpoints/mtl_mami.pt"
    )

    print("MAMI training completed & checkpoint saved")


if __name__ == "__main__":
    main()
