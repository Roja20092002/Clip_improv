import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from models.mtl_model import MTLModel
from datasets.loaders import get_mami_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------
    # Load model
    # ----------------------------
    model = MTLModel(num_tasks=6)
    checkpoint = torch.load("checkpoints/mtl_mami.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print("MAMI checkpoint loaded")

    # ----------------------------
    # DataLoader
    # ----------------------------
    loader = get_mami_loader(
        data_root="data/MAMI/data",
        split="validation",
        batch_size=8,
        transform=get_default_image_transform()
    )

    tokenizer = get_bert_tokenizer()

    tasks = ["label", "shaming", "stereotype", "objectification", "violence"]
    all_preds = {t: [] for t in tasks}
    all_labels = {t: [] for t in tasks}

    # ----------------------------
    # Validation Loop
    # ----------------------------
    for batch in tqdm(loader, desc="Validating MAMI"):
        images = batch["images"].to(device)
        texts = batch["texts"]
        labels = batch["labels"]
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

        for task in tasks:
            task_id = task_ids[task][0].item()

            with torch.no_grad():
                logits = model(
                    clip_inputs=clip_inputs,
                    uniter_inputs=clip_inputs,
                    bert_inputs=tokenized,
                    task_id=task_id
                )

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gt = labels[task].numpy()

            all_preds[task].extend(preds)
            all_labels[task].extend(gt)

    # ----------------------------
    # Metrics
    # ----------------------------
    print("\n===== MAMI FINAL EVALUATION =====")

    for task in tasks:
        acc = accuracy_score(all_labels[task], all_preds[task])
        cm = confusion_matrix(all_labels[task], all_preds[task])

        print(f"\nTask: {task}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        np.save(f"evaluation/{task}_confusion_matrix.npy", cm)

    print("\nAll confusion matrices saved in evaluation/ folder")


if __name__ == "__main__":
    main()
