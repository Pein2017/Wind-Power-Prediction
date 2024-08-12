import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from git import Optional

from models.layers.Embed import (
    PositionalEmbedding,
    TimeFeatureEmbedding,
    generate_x_mark,
)

# Create a custom logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

log_file = "mlp_v2.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both the handlers
formatter = logging.Formatter("%(asctime)s %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def adjust_channels_for_heads(channels, num_heads):
    # Find the nearest higher value that is divisible by num_heads and is even
    if channels % num_heads != 0 or channels % 2 != 0:
        channels = ((channels // num_heads) + 1) * num_heads
        if channels % 2 != 0:
            channels += num_heads  # Ensure it's even by adding num_heads
    return channels


def adjust_initial_input_dim(input_dim, num_heads):
    # Double the input_dim to calculate the out_channels
    out_channels = input_dim * 2

    # Adjust the out_channels to be divisible by num_heads and even
    adjusted_out_channels = adjust_channels_for_heads(out_channels, num_heads)

    # Adjust input_dim based on the adjusted_out_channels
    adjusted_input_dim = adjusted_out_channels // 2

    return adjusted_input_dim


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation_type=Optional[str],
    ):
        super(ConvLayer, self).__init__()

        if kernel_size < 4:
            raise ValueError(
                "kernel_size must be at least 4, otherwise it's uneffiecient."
            )

        if activation_type == "gelu":
            self.activation_type = nn.GELU()
        elif activation_type == "tanh":
            self.activation_type = nn.Tanh()
        elif activation_type == "relu":
            self.activation_type = nn.ReLU()
        else:
            self.activation_type = nn.Identity()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation_type(x)
        return x


class EnhancedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        seq_len,
        activation_type=None,
        fc_layer_type="mha",  # Optional values: 'mlp', 'mha', etc.
        num_heads=4,
        norm_type="batch",
        dropout=0.1,
        use_affine=True,
    ):
        super(EnhancedBlock, self).__init__()
        self.fc_layer_type = fc_layer_type
        self.num_heads = num_heads

        # Ensure out_channels is a multiple of num_heads
        out_channels = in_channels * 2
        adjusted_out_channels = adjust_channels_for_heads(out_channels, num_heads)

        # Conv1 and Conv2 for the block using the ConvLayer class
        self.conv1 = ConvLayer(
            in_channels,
            adjusted_out_channels,
            kernel_size,
            stride=1,
            padding=0,
            activation_type=activation_type,
        )
        self.conv2 = ConvLayer(
            adjusted_out_channels,
            adjusted_out_channels,
            kernel_size,
            stride=1,
            padding=0,
            activation_type=activation_type,
        )

        # Initial and adjusted sequence lengths after each convolution
        self.seq_len_after_conv1 = seq_len - (kernel_size - 1)
        self.seq_len_after_conv2 = self.seq_len_after_conv1 - (kernel_size - 1)

        # Residual connection with Conv1d to match dimensions
        self.residual_conv = nn.Conv1d(
            in_channels,
            adjusted_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Pooling layer to match the sequence length
        self.residual_pool = nn.AvgPool1d(
            kernel_size=(seq_len - self.seq_len_after_conv2 + 1), stride=1, padding=0
        )

        # Normalization layer
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(adjusted_out_channels, affine=use_affine)
            self.norm2 = nn.BatchNorm1d(adjusted_out_channels, affine=use_affine)
        else:
            self.norm1 = nn.LayerNorm(
                [self.seq_len_after_conv1, adjusted_out_channels],
                elementwise_affine=use_affine,
            )
            self.norm2 = nn.LayerNorm(
                [self.seq_len_after_conv2, adjusted_out_channels],
                elementwise_affine=use_affine,
            )

        # Choose between Attention or MLP or other layers
        if fc_layer_type == "mha":
            self.layer = nn.MultiheadAttention(
                adjusted_out_channels, num_heads=num_heads, batch_first=True
            )
        elif fc_layer_type == "mlp":
            self.layer = nn.Sequential(
                nn.Linear(adjusted_out_channels, adjusted_out_channels * 2),
                nn.GELU(),
                nn.Linear(adjusted_out_channels * 2, adjusted_out_channels),
            )
        else:
            raise ValueError(f"Unsupported layer_type: {fc_layer_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Permute input for Conv1d
        x = x.permute(
            0, 2, 1
        )  # [batch, seq_len, in_channels] -> [batch, in_channels, seq_len]

        # Residual path
        residual = self.residual_conv(x)  # Match the channel dimensions
        residual = self.residual_pool(residual)  # Match the sequence length

        # Convolution path - Conv1
        x = self.conv1(x)

        # Adjust LayerNorm after Conv1
        if self.norm_type == "batch":
            x = self.norm1(x)
        else:
            x = x.permute(
                0, 2, 1
            )  # [batch, in_channels, seq_len] -> [batch, seq_len, in_channels]
            x = self.norm1(x)
            x = x.permute(
                0, 2, 1
            )  # [batch, seq_len, in_channels] -> [batch, in_channels, seq_len]

        # Convolution path - Conv2
        x = self.conv2(x)

        # Adjust LayerNorm after Conv2
        if self.norm_type == "batch":
            x = self.norm2(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.norm2(x)
            x = x.permute(0, 2, 1)

        # Permute back to [batch, seq_len, num_features]
        x = x.permute(
            0, 2, 1
        )  # [batch, in_channels, seq_len] -> [batch, seq_len, in_channels]

        # Attention or MLP or other layers
        if self.fc_layer_type == "mha":
            x, _ = self.layer(x, x, x)
        else:
            x = self.layer(x)

        # Permute back for residual addition
        x = x.permute(
            0, 2, 1
        )  # [batch, seq_len, num_features] -> [batch, num_features, seq_len]

        # Add residual connection
        x = x + residual

        # Apply dropout and return to [batch, seq_len, out_channels]
        return self.dropout(x.permute(0, 2, 1))


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        time_features = ["hour"]  # ["hour", "quarter_hour", "day"]
        # Extract all necessary configuration values at the beginning
        original_input_dim = config.input_dim  # Original input dimension (e.g., 51)
        kernel_size = config.token_conv_kernel
        seq_len = config.seq_len
        num_heads = config.num_heads
        activation_type = config.activation_type
        fc_layer_type = config.fc_layer_type
        norm_type = config.norm_type
        dropout = config.dropout
        last_d_model = config.last_d_model
        pred_len = config.pred_len
        output_dim = config.output_dim
        use_pos_enc = config.use_pos_enc
        time_d_model = config.time_d_model
        combine_type = config.combine_type

        # Adjust the initial input_dim before any processing
        adjusted_input_dim = adjust_initial_input_dim(original_input_dim, num_heads)
        self.pred_len = pred_len
        self.output_dim = output_dim

        logger.debug(
            f"original_input_dim: {original_input_dim}, adjusted_input_dim: {adjusted_input_dim}"
        )

        # Adjust Conv1D to match the initial input_dim
        self.adjust_conv = nn.Conv1d(
            original_input_dim, adjusted_input_dim, kernel_size=1, stride=1, padding=0
        )

        # Initialize positional and temporal embeddings if needed
        self.positional_embedding = (
            PositionalEmbedding(adjusted_input_dim) if use_pos_enc else None
        )

        self.temporal_embedding = (
            TimeFeatureEmbedding(
                time_d_model=time_d_model,
                combine_type=combine_type,
                time_features=time_features,
                time_out_dim=adjusted_input_dim,  # Ensure time_out_dim matches adjusted input_dim
            )
            if time_features is not None
            else None
        )

        self.blocks = nn.ModuleList()
        current_seq_len = seq_len
        current_dim = adjusted_input_dim

        while current_seq_len - 2 * (kernel_size - 1) > 0:
            block = EnhancedBlock(
                in_channels=current_dim,
                kernel_size=kernel_size,
                seq_len=current_seq_len,
                activation_type=activation_type,
                fc_layer_type=fc_layer_type,
                num_heads=num_heads,
                norm_type=norm_type,
                dropout=dropout,
            )
            self.blocks.append(block)

            current_seq_len -= 2 * (kernel_size - 1)
            current_dim *= (
                2  # Update current_dim to the adjusted_dim for the next block
            )

        final_dim = current_dim
        self.mlp = nn.Sequential(
            nn.Linear(current_seq_len * final_dim, last_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(last_d_model, self.pred_len * self.output_dim),
        )

    def forward(
        self,
        x,
        x_mark=None,
        x_dec=None,
        x_dec_mark=None,
        time_decay=True,
        decay_factor=0.9,
    ):
        # Adjust the initial input dimension
        x = self.adjust_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply positional and temporal embeddings if applicable
        if self.positional_embedding is not None:
            pos_emb = self.positional_embedding(x)
            x += pos_emb

        if self.temporal_embedding is not None and x_mark is not None:
            time_emb = self.temporal_embedding(x_mark)
            x += time_emb  # Use addition since they have matching dimensions

        # Time-decaying mechanism
        if time_decay:
            decay_factors = torch.linspace(1.0, decay_factor, x.size(1)).to(x.device)
            x = x * decay_factors.unsqueeze(0).unsqueeze(-1)

        for block in self.blocks:
            x = block(x)

        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        x = x.view(x.size(0), self.pred_len, self.output_dim)
        return x


if __name__ == "__main__":
    import argparse

    device = torch.device("cuda")
    batch_size = 10

    """
    mlps are OK
    """
    test_settings = [
        # Odd input_dim, even kernel, even seq_len, odd num_heads, MHA
        {
            "input_dim": 51,
            "token_conv_kernel": 4,
            "seq_len": 32,
            "num_heads": 5,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Even input_dim, odd kernel, even seq_len, odd num_heads, MHA
        {
            "input_dim": 50,
            "token_conv_kernel": 5,
            "seq_len": 64,
            "num_heads": 7,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Odd input_dim, even kernel, odd seq_len, even num_heads, MHA
        {
            "input_dim": 53,
            "token_conv_kernel": 5,
            "seq_len": 31,
            "num_heads": 6,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Even input_dim, even kernel, odd seq_len, even num_heads, MHA
        {
            "input_dim": 64,
            "token_conv_kernel": 4,
            "seq_len": 29,
            "num_heads": 4,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Odd input_dim, odd kernel, even seq_len, even num_heads, MHA
        {
            "input_dim": 55,
            "token_conv_kernel": 7,
            "seq_len": 48,
            "num_heads": 8,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Even input_dim, odd kernel, odd seq_len, odd num_heads, MHA
        {
            "input_dim": 60,
            "token_conv_kernel": 6,
            "seq_len": 15,
            "num_heads": 5,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Odd input_dim, odd kernel, odd seq_len, even num_heads, MHA
        {
            "input_dim": 49,
            "token_conv_kernel": 11,
            "seq_len": 31,
            "num_heads": 6,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
        # Even input_dim, even kernel, even seq_len, odd num_heads, MHA
        {
            "input_dim": 62,
            "token_conv_kernel": 4,
            "seq_len": 32,
            "num_heads": 5,
            "norm_type": "layer",
            "fc_layer_type": "mha",
        },
    ]

    for idx, setting in enumerate(test_settings):
        print(f"Running test case {idx + 1} with settings: {setting}")

        # Define the configuration using argparse.Namespace
        config = argparse.Namespace(
            input_dim=setting["input_dim"],
            token_conv_kernel=setting["token_conv_kernel"],
            seq_len=setting["seq_len"],
            num_heads=setting["num_heads"],
            activation_type="gelu",
            fc_layer_type=setting["fc_layer_type"],
            norm_type=setting["norm_type"],
            dropout=0.1,
            pred_len=1,
            output_dim=1,
            use_pos_enc=False,
            time_features=["hour"],
            time_d_model=16,
            combine_type="add",
            last_d_model=16,
        )

        # Initialize the model with the config
        model = Model(config).to(device)

        print(model)

        # Create example inputs
        x = torch.randn(batch_size, config.seq_len, config.input_dim).to(
            device
        )  # [batch, seq_len, num_features]

        x_mark = generate_x_mark(batch_size, config.seq_len).to(device)

        # Forward pass
        output = model(x, x_mark)

        # Print the output tensor shape
        print(
            "Output tensor shape:", output.shape
        )  # Should print: [batch, pred_len, output_dim]
        print("-" * 50)
