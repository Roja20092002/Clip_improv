import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_clip: bool = False):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.embedding_dim = self.clip.config.projection_dim

        self.vision_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.text_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, input_ids, attention_mask, pixel_values):
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
            output_hidden_states=True,
        )

        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        vision_hidden_6 = vision_outputs.hidden_states[6]
        vision_hidden_12 = vision_outputs.hidden_states[12]
        text_hidden_6 = text_outputs.hidden_states[6]
        text_hidden_12 = text_outputs.hidden_states[12]

        vision_feature_6 = self.clip.visual_projection(
            self.clip.vision_model.post_layernorm(vision_hidden_6[:, 0, :])
        )
        vision_feature_12 = self.clip.visual_projection(
            self.clip.vision_model.post_layernorm(vision_hidden_12[:, 0, :])
        )

        text_feature_6 = self.clip.text_projection(
            self.clip.text_model.final_layer_norm(text_hidden_6[:, 0, :])
        )
        text_feature_12 = self.clip.text_projection(
            self.clip.text_model.final_layer_norm(text_hidden_12[:, 0, :])
        )

        vision_alpha = torch.softmax(self.vision_logits, dim=0)
        text_alpha = torch.softmax(self.text_logits, dim=0)

        image_features = vision_alpha[0] * vision_feature_6 + vision_alpha[1] * vision_feature_12
        text_features = text_alpha[0] * text_feature_6 + text_alpha[1] * text_feature_12

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        return image_features, text_features