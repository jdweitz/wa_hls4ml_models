# models.py

import torch
import torch.nn as nn

class LayerTokenizer(nn.Module):
    def __init__(self, feature_dim: int = 34, embed_dim: int = 512, max_layers: int = 128):
        """
        Maps per-layer features to token embeddings with learned positional embeddings and a [CLS] token.
        """
        super().__init__()
        self.linear = nn.Linear(feature_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(max_layers, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, feature_dim), where L <= max_layers
        """
        B, L, _ = x.shape
        tokens = self.linear(x)  # (B, L, embed_dim)
        tokens = tokens + self.pos_emb[:L, :].unsqueeze(0)  # positional embedding
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, L+1, embed_dim)
        return tokens

class TransformerRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 34, #changed from 32, bc now have 2 more features
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_layers: int = 2,
        output_dim: int = 6,
        max_layers: int = 51, # changed from 18
        dropout: float = 0.1,
    ):
        """
        Transformer based regressor for variable length input sequences.
        """
        super().__init__()
        self.tokenizer = LayerTokenizer(feature_dim, embed_dim, max_layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x, pad_mask):
        """
        x:        (B, L, feature_dim)
        pad_mask: (B, L) True=padding
        returns:
          preds:  (B, output_dim)
        """
        B, L, _ = x.shape
        tokens = self.tokenizer(x)  # (B, L+1, embed_dim)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_pad, pad_mask], dim=1)  # (B, L+1)
        tokens = tokens.transpose(0, 1)  # (L+1, B, embed_dim)
        out = self.transformer(tokens, src_key_padding_mask=mask)
        cls_out = out[0]  # (B, embed_dim)
        return self.head(cls_out)  # (B, output_dim)
    
## To use:
# from models import TransformerRegressor

# model = TransformerRegressor(
#     feature_dim=16,
#     embed_dim=512,
#     num_heads=8,
#     ff_dim=512,
#     num_layers=2,
#     output_dim=6,
#     max_layers=18,
#     dropout=0.1
# )