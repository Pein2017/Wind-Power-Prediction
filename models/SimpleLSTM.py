import torch
import torch.nn as nn

from models.layers.Embed import (
    CrossChannelBlock,
    TemporalFeatureEmbedding,
    TokenEmbedding,
)


class SequenceBlock(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False
    ):
        super(SequenceBlock, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only between layers
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim * num_directions]
        if self.bidirectional:
            lstm_out = (
                lstm_out[:, :, : self.hidden_dim] + lstm_out[:, :, self.hidden_dim :]
            )  # Sum bidirectional outputs
        return lstm_out


class CombinedBlock(nn.Module):
    def __init__(
        self,
        seq_input_dim,
        cross_input_dim,
        hidden_dim,
        num_layers,
        dropout=0.1,
        norm_type="batch",
        bidirectional=False,
    ):
        super(CombinedBlock, self).__init__()
        self.sequence_block = SequenceBlock(
            seq_input_dim, hidden_dim, num_layers, dropout, bidirectional
        )
        self.cross_channel_block = CrossChannelBlock(
            hidden_dim + cross_input_dim, hidden_dim, dropout
        )
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        seq_out = self.sequence_block(x)  # [batch, seq_len, hidden_dim]
        if x_mark.size(-1) > 0:
            combined = torch.cat(
                (seq_out, x_mark), dim=-1
            )  # [batch, seq_len, hidden_dim + cross_input_dim]
        else:
            combined = seq_out  # If no time features, skip concatenation
        cross_out = self.cross_channel_block(combined)  # [batch, seq_len, hidden_dim]
        if self.norm_type == "batch":
            cross_out = self.norm(cross_out.transpose(1, 2)).transpose(
                1, 2
            )  # Batch normalization
        elif self.norm_type == "layer":
            cross_out = self.norm(cross_out)  # Layer normalization
        cross_out = self.dropout(cross_out)
        return cross_out + seq_out  # Skip connection


class Model(nn.Module):
    def __init__(self, config, time_features=[]):
        super(Model, self).__init__()

        input_dim = config.input_dim
        d_model = config.d_model
        time_d_model = config.time_d_model
        hidden_dim = config.hidden_dim
        seq_layers = config.seq_layers
        e_layers = config.e_layers
        dropout = config.dropout
        bidirectional = config.bidirectional
        norm_type = config.norm_type  # "batch" or "layer"

        num_time_features = len(time_features)

        self.token_embedding = TokenEmbedding(input_dim, d_model)
        self.time_embedding = (
            TemporalFeatureEmbedding(time_d_model, time_features=time_features)
            if num_time_features > 0
            else None
        )
        self.combined_blocks = nn.ModuleList()

        # Initialize the first CombinedBlock
        self.combined_blocks.append(
            CombinedBlock(
                input_dim * d_model,
                num_time_features * time_d_model if num_time_features > 0 else 0,
                hidden_dim,
                seq_layers,
                dropout,
                norm_type,
                bidirectional,
            )
        )

        # Initialize subsequent CombinedBlocks
        for _ in range(1, e_layers):
            self.combined_blocks.append(
                CombinedBlock(
                    hidden_dim,
                    num_time_features * time_d_model if num_time_features > 0 else 0,
                    hidden_dim,
                    seq_layers,
                    dropout,
                    norm_type,
                    bidirectional,
                )
            )

        self.final_dense = nn.Linear(hidden_dim, 1)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None, x_dec=None, x_dec_mark=None):
        x_emb = self.token_embedding(x)  # [batch, seq_len, input_dim * d_model]
        if self.time_embedding is not None:
            x_mark_emb = self.time_embedding(
                x_mark
            )  # [batch, seq_len, num_time_features * time_d_model]
        else:
            x_mark_emb = torch.empty(
                x_emb.size(0), x_emb.size(1), 0, device=x_emb.device
            )  # Empty tensor for x_mark_emb

        for block in self.combined_blocks:
            x_emb = block(x_emb, x_mark_emb)

        x_emb = self.final_dropout(x_emb[:, -1, :])  # Apply final dropout
        output = self.final_dense(x_emb).unsqueeze(1)  # [batch, 1, 1]
        return output
