import torch
import torch.nn as nn

from models.clip_encoder import CLIPEncoder


class CLIPBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = CLIPEncoder()

        embedding_dim = self.encoder.embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512), # type: ignore
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, pixel_values):

        image_features, text_features = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        features = torch.cat((image_features, text_features), dim=1)

        logits = self.classifier(features)

        return logits