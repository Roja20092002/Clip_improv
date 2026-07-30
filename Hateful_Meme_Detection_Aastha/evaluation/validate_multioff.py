import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

from models.mtl_model import MTLModel
from datasets.loaders import get_multioff_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = MTLModel(num_tasks=7)  # includes MultiOFF
    model.to(device)
    model.eval()

    # Load checkpoint (saved as state_dict only)
    state_dict = torch.load("checkpoints/mtl_multioff.pt", map_location=device)
    model.load_state_dict(state_dict)

    print("MultiOFF checkpoint loaded")

    # --------------------------------------------------
    # Data loader
    # --------------------------------------------------
    loader = get_multioff_loader(
        data_root="data/MultiOFF",
        split="test",
        batch_size=1,   # MultiOFF single-sample style
        transform=get_default_image_transform()
    )

    tokenizer = get_bert_tokenizer()

    all_preds = []
    all_labels = []

    # --------------------------------------------------
    # Validation loop
    # --------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating MultiOFF"):
            images = batch["image"].to(device)     # [B, 3, 224, 224]
            texts = batch["text"]                  # list of strings
            labels = batch["labels"].to(device)    # [B]

            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

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
                task_id=6   # MultiOFF task ID
            )

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== MULTIOFF VALIDATION RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    np.save("evaluation/multioff_confusion_matrix.npy", cm)
    print("Confusion matrix saved")


if __name__ == "__main__":
    main()
