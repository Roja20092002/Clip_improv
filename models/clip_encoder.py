import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_clip: bool = False):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)

        # Optionally freeze the CLIP backbone parameters so they are not updated during training.
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.embedding_dim = self.clip.config.projection_dim

    def forward(self, input_ids, attention_mask, pixel_values):

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )

        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        return image_features, text_features