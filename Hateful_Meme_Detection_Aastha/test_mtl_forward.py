import torch
from PIL import Image

# Models
from models.mtl_model import MTLModel

# Tokenizers / processors
from utils.text_tokenizers import get_bert_tokenizer
from utils.clip_processors import get_clip_image_processor
from transformers import CLIPTokenizer

# -----------------------------
# 1. Load model
# -----------------------------
model = MTLModel(num_tasks=4)
model.eval()

# -----------------------------
# 2. Load tokenizers / processors
# -----------------------------
bert_tokenizer = get_bert_tokenizer()
clip_image_processor = get_clip_image_processor()
clip_tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# -----------------------------
# 3. Sample input
# -----------------------------
text = "white people is this a shooting range"
image = Image.open("data/HatefulMemes/data/img/08291.png")
# -----------------------------
# BERT inputs (for BERT + UNITER)
# -----------------------------
bert_inputs = bert_tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=32,
    return_tensors="pt"
)


# -----------------------------
# 4. CLIP inputs (CORRECT WAY)
# -----------------------------
clip_text = clip_tokenizer(
    text,
    return_tensors="pt",
    padding=True
)

clip_image = clip_image_processor(
    images=image,
    return_tensors="pt"
)

clip_inputs = {
    "input_ids": clip_text["input_ids"],
    "attention_mask": clip_text["attention_mask"],
    "pixel_values": clip_image["pixel_values"]
}

# -----------------------------
# 5. UNITER inputs
# -----------------------------
# -----------------------------
# 5. UNITER inputs (USE BERT TOKENS)
# -----------------------------
uniter_inputs = {
    "input_ids": bert_inputs["input_ids"],
    "attention_mask": bert_inputs["attention_mask"],
    "pixel_values": clip_image["pixel_values"]
}


# -----------------------------
# 6. BERT inputs
# -----------------------------
bert_inputs = bert_tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

# -----------------------------
# 7. Forward pass (task_id = 0)
# -----------------------------
with torch.no_grad():
    logits = model(
        clip_inputs=clip_inputs,
        uniter_inputs=uniter_inputs,
        bert_inputs=bert_inputs,
        task_id=0
    )

# -----------------------------
# 8. Output
# -----------------------------
print("MTL logits shape:", logits.shape)
print("MTL logits:", logits)
