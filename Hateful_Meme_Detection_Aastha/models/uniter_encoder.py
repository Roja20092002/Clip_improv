import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

class UNITEREncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

        self.text_dim = self.text_encoder.config.hidden_size
        self.image_dim = self.image_encoder.config.hidden_size

        self.fusion = nn.Linear(self.text_dim + self.image_dim, 768)
        self.output_dim = 768

    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        image_out = self.image_encoder(
            pixel_values=pixel_values
        ).pooler_output

        fused = torch.cat([text_out, image_out], dim=1)
        return self.fusion(fused)
