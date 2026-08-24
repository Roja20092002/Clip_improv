import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_clip: bool = False, unfreeze_layers: int = 0):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)
        self.unfreeze_layers = max(0, min(3, int(unfreeze_layers)))

        # Compatibility with older frozen-backbone runs.
        if freeze_clip:
            self.unfreeze_layers = 0

        # Freeze the whole CLIP backbone first.
        for param in self.clip.parameters():
            param.requires_grad = False

        # Unfreeze only the selected final transformer layers in both branches.
        if self.unfreeze_layers > 0:
            start_layer = 12 - self.unfreeze_layers
            for layer_idx in range(start_layer, 12):
                for param in self.clip.vision_model.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True
                for param in self.clip.text_model.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

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