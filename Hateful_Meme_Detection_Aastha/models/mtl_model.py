import torch
import torch.nn as nn

from models.clip_encoder import CLIPEncoder
from models.uniter_encoder import UNITEREncoder
from transformers import BertModel


class MTLModel(nn.Module):
    def __init__(self, num_tasks=8):
        super().__init__()

        # Encoders
        self.clip = CLIPEncoder()
        self.uniter = UNITEREncoder()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Dimensions
        self.clip_dim = self.clip.output_dim
        self.uniter_dim = self.uniter.output_dim
        self.bert_dim = self.bert.config.hidden_size

        fused_dim = self.clip_dim + self.uniter_dim + self.bert_dim

        # Shared fusion layer
        self.shared_fc = nn.Linear(fused_dim, 512)
        self.relu = nn.ReLU()

        # Task-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(512, 2) for _ in range(num_tasks)
        ])

    def forward(
        self,
        clip_inputs,
        uniter_inputs,
        bert_inputs,
        task_id
    ):
        clip_feat = self.clip(
            input_ids=clip_inputs["input_ids"],
            attention_mask=clip_inputs["attention_mask"],
            pixel_values=clip_inputs["pixel_values"]
        )

        uniter_feat = self.uniter(**uniter_inputs)

        bert_feat = self.bert(
            input_ids=bert_inputs["input_ids"],
            attention_mask=bert_inputs["attention_mask"]
        ).pooler_output

        fused = torch.cat(
            [clip_feat, uniter_feat, bert_feat],
            dim=1
        )

        shared = self.relu(self.shared_fc(fused))
        logits = self.heads[task_id](shared)

        return logits
