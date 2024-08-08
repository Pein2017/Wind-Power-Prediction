import torch
import torch.nn as nn

from models.layers.Embed import (
    CrossChannelBlock,
    PositionalEmbedding,
    TemporalFeatureEmbedding,
    TokenEmbedding,
)


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, hidden_dim, dropout=0.1, norm_type="batch"
    ):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            self.norm1 = nn.BatchNorm1d(embed_dim)
            self.norm2 = nn.BatchNorm1d(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.norm1(x + attn_out)
        else:
            x = self.norm1((x + attn_out).transpose(1, 2)).transpose(1, 2)

        ffn_out = self.ffn(x)
        if isinstance(self.norm2, nn.LayerNorm):
            x = self.norm2(x + ffn_out)
        else:
            x = self.norm2((x + ffn_out).transpose(1, 2)).transpose(1, 2)

        return x


class CombinedBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_dim,
        dropout=0.1,
        norm_type="batch",
        cross_channel_dim=None,
    ):
        super(CombinedBlock, self).__init__()
        self.transformer_block = TransformerBlock(
            embed_dim, num_heads, hidden_dim, dropout, norm_type
        )
        self.cross_channel_block = (
            CrossChannelBlock(embed_dim + cross_channel_dim, hidden_dim, dropout)
            if cross_channel_dim
            else None
        )

        if norm_type == "layer":
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = nn.BatchNorm1d(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.transformer_block(x)
        if self.cross_channel_block:
            if x_mark is not None and x_mark.size(-1) > 0:
                combined = torch.cat(
                    (x, x_mark), dim=-1
                )  # [batch, seq_len, embed_dim + time_d_model]
            else:
                combined = x  # If no time features, use x only
            cross_out = self.cross_channel_block(
                combined
            )  # [batch, seq_len, embed_dim]

            if isinstance(self.norm, nn.LayerNorm):
                cross_out = self.norm(cross_out)
            else:
                cross_out = self.norm(cross_out.transpose(1, 2)).transpose(
                    1, 2
                )  # Normalization

            cross_out = self.dropout(cross_out)
            return cross_out + x  # Skip connection
        return x


class Model(nn.Module):
    def __init__(self, config, time_features=[]):
        super(Model, self).__init__()

        input_dim = config.input_dim
        d_model = config.d_model  # dimension per input feature
        embed_dim = input_dim * d_model  # total embedding dimension
        num_heads = config.num_heads
        hidden_dim = config.hidden_dim
        num_layers = config.e_layers
        dropout = config.dropout
        norm_type = config.norm_type  # "batch" or "layer"

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        time_d_model = config.time_d_model
        num_time_features = len(time_features)

        self.token_embedding = TokenEmbedding(input_dim, d_model)
        self.positional_encoding = (
            PositionalEmbedding(embed_dim) if config.use_pos_enc else None
        )
        self.time_embedding = (
            TemporalFeatureEmbedding(time_d_model, time_features=time_features)
            if num_time_features > 0
            else None
        )

        self.blocks = nn.ModuleList(
            [
                CombinedBlock(
                    embed_dim,
                    num_heads,
                    hidden_dim,
                    dropout,
                    norm_type,
                    cross_channel_dim=time_d_model if num_time_features > 0 else None,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_dense = nn.Linear(embed_dim, 1)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None, x_dec=None, x_dec_mark=None):
        x_emb = self.token_embedding(x)  # [batch, seq_len, embed_dim]
        if self.positional_encoding is not None:
            x_emb = x_emb + self.positional_encoding(x_emb)

        if self.time_embedding is not None and x_mark is not None:
            x_mark_emb = self.time_embedding(x_mark)  # [batch, seq_len, time_d_model]
        else:
            x_mark_emb = torch.empty(
                x_emb.size(0), x_emb.size(1), 0, device=x_emb.device
            )  # Empty tensor for x_mark_emb

        for block in self.blocks:
            x_emb = block(x_emb, x_mark_emb)

        x_emb = self.final_dropout(x_emb[:, -1, :])  # Apply final dropout
        output = self.final_dense(x_emb).unsqueeze(1)  # [batch, 1, 1]
        return output
