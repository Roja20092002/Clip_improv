import os
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoProcessor

from models.clip_baseline import CLIPBaseline
from data_loaders.mami_dataset import MAMIDataset

from utils.metrics import calculate_metrics
from utils.seed import set_seed
import configs.config as config
from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = config.MODEL_NAME

BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS

SEED = config.SEED

# Persistent Google Drive location
CHECKPOINT_DIR = os.environ.get(
    "CLIP_CHECKPOINT_DIR",
    "/content/drive/MyDrive/Clip_improv_experiments/mami_clip_baseline"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

LAST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_checkpoint.pth"
)

FINAL_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "final_model.pth"
)

HISTORY_PATH = os.path.join(
    CHECKPOINT_DIR,
    "history.json"
)

CONFIG_PATH = os.path.join(
    CHECKPOINT_DIR,
    "config.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CLIP MAMI BASELINE TRAINING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

print("=" * 60)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

experiment_config = {
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "seed": SEED,
    "device": str(device),
}

with open(CONFIG_PATH, "w") as f:
    json.dump(experiment_config, f, indent=4)


# ============================================================
# LOAD MAMI
# ============================================================

print("\nLoading MAMI dataset...")

hf_dataset = load_dataset(
    "scintist/MAMI-dataset"
)

full_dataset = hf_dataset["train"]

print(
    "Total samples:",
    len(full_dataset)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

indices = list(range(len(full_dataset)))

labels = [
    0 if label == 0 else 1
    for label in full_dataset["multiclass_label"]
]

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.1,
    random_state=SEED,
    stratify=labels,
)

train_hf = full_dataset.select(train_indices)
val_hf = full_dataset.select(val_indices)

print("Training samples:", len(train_hf))
print("Validation samples:", len(val_hf))


# ============================================================
# PROCESSOR
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = MAMIDataset(
    train_hf,
    processor
)

val_dataset = MAMIDataset(
    val_hf,
    processor
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# MODEL
# ============================================================

print("\nLoading CLIP model...")

model = CLIPBaseline(
    num_classes=config.NUM_CLASSES,
    freeze_clip=True,
).to(device)

# Temporary sanity check: print parameter counts by trainability.
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"Model parameters — total: {total_params}, trainable: {trainable_params}, frozen: {frozen_params}")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# RESUME
# ============================================================

start_epoch = 0
best_f1 = -1.0
history = []

if os.path.exists(LAST_CHECKPOINT_PATH):

    print("\nExisting checkpoint found.")
    print("Resuming training...")

    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if checkpoint.get("scaler_state_dict") is not None:

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = checkpoint["epoch"] + 1

    best_f1 = checkpoint["best_f1"]

    history = checkpoint.get(
        "history",
        []
    )

    print(
        f"Resuming from epoch {start_epoch + 1}"
    )

    print(
        f"Previous best F1: {best_f1:.4f}"
    )


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping_patience = 3
epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    start_epoch,
    EPOCHS
):

    print("\n")
    print("=" * 60)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_labels = []

    progress = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch + 1}",
        unit="batch"
    )

    for batch in progress:

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True
        )

        pixel_values = batch[
            "pixel_values"
        ].to(
            device,
            non_blocking=True
        )

        target = batch[
            "label"
        ].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------
        # Forward pass
        # --------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

            loss = criterion(
                logits,
                target
            )

        # --------------------------------------------
        # Backward pass
        # --------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        # --------------------------------------------
        # Statistics
        # --------------------------------------------

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_labels.extend(
            target.detach()
            .cpu()
            .numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    train_loss = (
        running_loss
        / len(train_loader)
    )

    train_metrics = calculate_metrics(
        train_labels,
        train_predictions
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_predictions = []
    val_labels = []

    progress = tqdm(
        val_loader,
        desc=f"Validation Epoch {epoch + 1}",
        unit="batch"
    )

    with torch.no_grad():

        for batch in progress:

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                device,
                non_blocking=True
            )

            pixel_values = batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True
            )

            target = batch[
                "label"
            ].to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

                loss = criterion(
                    logits,
                    target
                )

            val_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                target.cpu().numpy()
            )

    val_loss /= len(val_loader)

    val_metrics = calculate_metrics(
        val_labels,
        val_predictions
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_metrics["f1"]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    current_lr = optimizer.param_groups[0]["lr"]

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Learning Rate : {current_lr:.2e}"
    )

    print(
        f"Train Loss    : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_metrics['accuracy']:.4f}"
    )

    print(
        f"Train F1      : {train_metrics['f1']:.4f}"
    )

    print(
        f"Val Loss      : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy  : {val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Precision : {val_metrics['precision']:.4f}"
    )

    print(
        f"Val Recall    : {val_metrics['recall']:.4f}"
    )

    print(
        f"Val F1        : {val_metrics['f1']:.4f}"
    )

    print("-" * 60)


    # ========================================================
    # HISTORY
    # ========================================================

    epoch_record = {

        "epoch": epoch + 1,

        "learning_rate": current_lr,

        "train_loss": train_loss,

        "train_accuracy":
            train_metrics["accuracy"],

        "train_precision":
            train_metrics["precision"],

        "train_recall":
            train_metrics["recall"],

        "train_f1":
            train_metrics["f1"],

        "val_loss": val_loss,

        "val_accuracy":
            val_metrics["accuracy"],

        "val_precision":
            val_metrics["precision"],

        "val_recall":
            val_metrics["recall"],

        "val_f1":
            val_metrics["f1"],
    }

    history.append(
        epoch_record
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"\n✓ NEW BEST MODEL!"
            f" Val F1 = {best_f1:.4f}"
        )

        print(
            "Saved:",
            BEST_MODEL_PATH
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement."
            f" Patience:"
            f" {epochs_without_improvement}/"
            f"{early_stopping_patience}"
        )


    # ========================================================
    # SAVE FULL CHECKPOINT
    # ========================================================

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict()
                if use_amp
                else None,

            "best_f1":
                best_f1,

            "history":
                history,

            "config":
                experiment_config,
        },
        LAST_CHECKPOINT_PATH
    )

    print(
        "✓ Checkpoint saved:"
    )

    print(
        LAST_CHECKPOINT_PATH
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= early_stopping_patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation F1: {best_f1:.4f}"
)

print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "Final model:",
    FINAL_MODEL_PATH
)

print(
    "History:",
    HISTORY_PATH
)