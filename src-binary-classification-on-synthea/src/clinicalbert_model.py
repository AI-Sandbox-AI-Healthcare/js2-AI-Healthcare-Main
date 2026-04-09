import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ClinicalBERT_Transformer(nn.Module):
    def __init__(self, bert_model, structured_input_dim=None,
                 hidden_dim=128, nhead=8, num_layers=2,
                 dropout=0.3):
        super().__init__()

        self.bert = bert_model

        # Project CLS embeddings
        self.bert_proj = nn.Linear(768, hidden_dim)

        # Structured projection
        if structured_input_dim is not None:
            self.struct_proj = nn.Linear(structured_input_dim, hidden_dim)
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.struct_proj = None
            self.fusion_proj = None

        # Transformer across visits
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Single logit for binary classification
        )

    def forward(self, input_ids, attention_mask, structured_seq=None, visit_mask=None):
        """
        input_ids: (B, T, L) 
        attention_mask: (B, T, L)
        structured_seq: (B, T, F) or None
        visit_mask: (B, T)
        """
        B, T, L = input_ids.shape

        input_ids = input_ids.view(B * T, L)
        attention_mask = attention_mask.view(B * T, L)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0, :]
        bert_emb = cls.view(B, T, 768)

        bert_emb = self.bert_proj(bert_emb)  # (B, T, H)

        if self.struct_proj is not None and structured_seq is not None:
            struct_emb = self.struct_proj(structured_seq)  # (B, T, H)
            fused = torch.cat([bert_emb, struct_emb], dim=-1)
            fused = self.fusion_proj(fused)
        else:
            fused = bert_emb

        if visit_mask is not None:
            key_padding_mask = ~visit_mask.bool()
            tr_out = self.transformer_encoder(fused, src_key_padding_mask=key_padding_mask)
            mask = visit_mask.unsqueeze(-1).to(tr_out.device).float()
            pooled = (tr_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            tr_out = self.transformer_encoder(fused)
            pooled = tr_out.mean(dim=1)

        return self.classifier(pooled)
