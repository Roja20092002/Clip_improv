import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

from datasets.asian_hate_dataset import AsianHateDataset
from models.mtl_model import MTLModel
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = MTLModel(num_tasks=5)
    model.load_state_dict(torch.load("checkpoints/mtl_joint.pt", map_location=device))
    model.to(device)
    model.eval()

    print("Joint MTL model loaded")

    # Load dataset
    dataset = AsianHateDataset(
        data_root="data/Asian_Hate",
        split="test",
        transform=get_default_image_transform()
    )

    tokenizer = get_bert_tokenizer()

    correct = 0
    total = 0

    # For confusion matrix
    all_preds = []
    all_labels = []

    for sample in tqdm(dataset, desc="Evaluating Asian Hate Dataset"):
        image = sample["image"].unsqueeze(0).to(device)
        text = [sample["text"]]
        label = sample["label"].to(device)

        # Tokenization (fixed)
        tokenized = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # Forward pass
        with torch.no_grad():
            logits = model(
                clip_inputs={
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "pixel_values": image
                },
                uniter_inputs={
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "pixel_values": image
                },
                bert_inputs=tokenized,
                task_id=4   # ✅ asian_hate
            )

        pred = torch.argmax(logits, dim=1)

        # Accuracy
        if pred.item() == label.item():
            correct += 1

        total += 1

        # ✅ Store for confusion matrix
        all_preds.append(pred.item())
        all_labels.append(label.item())

    accuracy = correct / total if total > 0 else 0

    print("\n===== ASIAN_HATE RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")

    # ✅ Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)

    # ✅ Save matrix
    np.save("evaluation/asian_hate_confusion_matrix.npy", cm)
    print("Confusion matrix saved as asian_hate_confusion_matrix.npy")


if __name__ == "__main__":
    main()