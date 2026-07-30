import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        self.output_dim = self.clip.config.projection_dim

    def forward(self, input_ids, attention_mask, pixel_values):

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )

        return {
            "image": outputs.image_embeds,
            "text": outputs.text_embeds
        }