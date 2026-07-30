import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models.mtl_model import MTLModel
from datasets.loaders import get_hateful_memes_loader
from utils.image_transforms import get_default_image_transform
from utils.text_tokenizers import get_bert_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load JOINT model
    model = MTLModel(num_tasks=8)
    model.load_state_dict(torch.load("checkpoints/mtl_joint.pt", map_location=device))
    model.to(device)
    model.eval()

    print("Joint MTL model loaded")

    loader = get_hateful_memes_loader(
        data_root="data/HatefulMemes/data",
        split="dev",
        batch_size=8,
        transform=get_default_image_transform(),
        shuffle=False
    )

    tokenizer = get_bert_tokenizer()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Joint on Hateful Memes"):
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
                task_id=0  # Hateful Memes
            )

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nJoint Model Accuracy on Hateful Memes: {acc:.4f}")


if __name__ == "__main__":
    main()
