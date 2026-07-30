import torch
import torch.nn as nn
from torch.optim import AdamW

from models.mtl_model import MTLModel
from datasets.loaders import get_joint_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ✅ Model
    model = MTLModel(num_tasks=5)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # ✅ Loader
    loader = get_joint_loader(
        transform=get_default_image_transform(),
        batch_size=8
    )

    tokenizer = get_bert_tokenizer()

    # ✅ Training settings
    num_steps = 30000
    step = 0

    # ✅ FIX: step-based training loop
    while step < num_steps:
        for batch in loader:

            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)
            task_ids = batch["task_ids"]

            # ✅ Tokenization (improved)
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            optimizer.zero_grad()

            logits = model(
                clip_inputs={
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "pixel_values": images
                },
                uniter_inputs={
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "pixel_values": images
                },
                bert_inputs=tokenized,
                task_id=task_ids[0].item()
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step} | Loss {loss.item():.4f}")

            step += 1

            if step >= num_steps:
                break

    # ✅ Save model
    torch.save(model.state_dict(), "checkpoints/mtl_joint.pt")
    print("Joint MTL training finished")


if __name__ == "__main__":
    main()