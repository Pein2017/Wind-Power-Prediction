import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation=None,
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class EnhancedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        seq_len,
        activation=None,
        use_attention=False,
        num_heads=4,
        norm_type="batch",
        dropout=0.1,
    ):
        super(EnhancedBlock, self).__init__()
        self.use_attention = use_attention
        out_channels = in_channels * 2

        # Conv1 and Conv2 for the block
        self.conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            activation=activation,
        )
        self.conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            activation=activation,
        )

        # Adjusted sequence length after convolutions
        self.adjusted_seq_len = seq_len - 2 * (kernel_size - 1)

        # Residual connection with Conv1d to match dimensions
        self.residual_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.residual_pool = nn.AvgPool1d(
            kernel_size=2 * (kernel_size - 1) + 1,
            stride=1,
            padding=0,
            count_include_pad=False,
        )

        # Normalization layer
        if norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        else:
            self.norm1 = nn.LayerNorm([self.adjusted_seq_len, out_channels])
            self.norm2 = nn.LayerNorm([self.adjusted_seq_len, out_channels])

        # Attention or MLP layer
        if use_attention:
            self.attention = nn.MultiheadAttention(
                out_channels, num_heads=num_heads, batch_first=True
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels * 2),
                nn.GELU(),
                nn.Linear(out_channels * 2, out_channels),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Residual path
        residual = self.residual_conv(
            x.permute(0, 2, 1)
        )  # Permute to [batch, num_features, seq_len] for Conv1d
        residual = self.residual_pool(residual)

        # Convolution path
        x = self.conv1(x.permute(0, 2, 1))
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm1(x)

        x = self.conv2(x)
        if isinstance(self.norm2, nn.LayerNorm):
            x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm2(x)

        # Attention or MLP path
        x = x.permute(
            0, 2, 1
        )  # Permute to [batch, seq_len, num_features] for Attention or MLP
        if self.use_attention:
            x, _ = self.attention(x, x, x)
        else:
            x = self.mlp(x)

        # Final path
        x = x.permute(
            0, 2, 1
        )  # Permute back to [batch, num_features, seq_len] for Conv1d
        x = self.dropout(x)

        x = x + residual  # Add residual connection
        return x.permute(0, 2, 1)  # Permute back to [batch, seq_len, num_features]


class ForecastingModel(nn.Module):
    def __init__(
        self,
        input_dim,
        kernel_size,
        seq_len,
        e_layers,
        use_attention=False,
        num_heads=4,
        activation=nn.GELU(),
        norm_type="batch",
        dropout=0.1,
        pred_len=1,
        output_dim=1,
    ):
        super(ForecastingModel, self).__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.blocks = nn.ModuleList()
        current_seq_len = seq_len
        current_dim = input_dim

        for i in range(e_layers):
            if current_seq_len - 2 * (kernel_size - 1) <= 0:
                print("Error: Sequence length too small for kernel size.")
                break

            block = EnhancedBlock(
                current_dim,
                kernel_size,
                current_seq_len,
                activation=activation,
                use_attention=use_attention,
                num_heads=num_heads,
                norm_type=norm_type,
                dropout=dropout,
            )
            self.blocks.append(block)

            current_seq_len -= 2 * (kernel_size - 1)
            current_dim *= 2

        final_dim = current_dim
        self.mlp = nn.Sequential(
            nn.Linear(current_seq_len * final_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.pred_len * self.output_dim),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        x = x.view(x.size(0), self.pred_len, self.output_dim)
        return x


if __name__ == "__main__":
    # Fix input_dim
    input_dim = 51

    # Define different feasible kernel_size and seq_len combinations
    settings = [
        {"kernel_size": 3, "seq_len": 32, "num_heads": 3},
        {"kernel_size": 5, "seq_len": 64, "num_heads": 6},
        {"kernel_size": 4, "seq_len": 128, "num_heads": 17},
        {"kernel_size": 6, "seq_len": 50, "num_heads": 2},
        {"kernel_size": 7, "seq_len": 25, "num_heads": 6},
    ]

    for setting in settings:
        kernel_size = setting["kernel_size"]
        seq_len = setting["seq_len"]
        num_heads = setting["num_heads"]

        model = ForecastingModel(
            input_dim,
            kernel_size,
            seq_len,
            e_layers=3,
            use_attention=True,
            num_heads=num_heads,
            activation=nn.GELU(),
            norm_type="batch",
            dropout=0.1,
            pred_len=1,
            output_dim=1,
        ).cuda()  # Move model to CUDA

        x = torch.randn(
            10, seq_len, input_dim
        ).cuda()  # Example input: [batch, seq_len, num_features]
        output = model(x)
        print(
            f"Settings: kernel_size={kernel_size}, seq_len={seq_len}, num_heads={num_heads}"
        )
        print(
            "Output tensor shape:", output.shape
        )  # Should print the output tensor shape
        print("-" * 50)
