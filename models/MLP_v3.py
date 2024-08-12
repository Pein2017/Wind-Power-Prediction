# noqa
import logging

import torch
import torch.nn as nn

from models.layers.Embed import (
    PositionalEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
    generate_x_mark,
)

# Create a custom logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

log_file = "mlp_v3.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both the handlers
formatter = logging.Formatter("%(asctime)s %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_activation(activation_type: str) -> nn.Module:
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type is None:
        return nn.Identity()  # No activation_type
    else:
        raise ValueError(f"Unsupported activation_type type: {activation_type}")


class InitBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        token_d_model,
        pos_d_model,
        time_d_model,
        conv_out_dim,
        time_features=None,
        token_conv_kernel=5,
    ):
        super(InitBlock, self).__init__()

        if token_conv_kernel < 4:
            raise ValueError(
                "token_conv_kernel should be at least 4 for high efficiency"
            )

        # Token embedding: Project each input feature to a higher dimension
        self.token_embedding = TokenEmbedding(input_dim, token_d_model)

        # Positional and Temporal Embedding
        self.positional_embedding = PositionalEmbedding(pos_d_model)
        self.temporal_embedding = TimeFeatureEmbedding(
            time_d_model, time_features=time_features
        )

        # Conv1D layer
        total_dim = input_dim * token_d_model + pos_d_model + time_d_model
        self.conv1d = nn.Conv1d(
            total_dim,
            conv_out_dim,
            kernel_size=token_conv_kernel,
            padding=(token_conv_kernel - 1) // 2,  # Ensure output seq_len matches input
            padding_mode="zeros",
        )

    def forward(self, x, x_mark=None):
        batch_size, seq_len, _ = x.size()

        # Token embedding
        X_token_out = self.token_embedding(x)

        # Manually create Positional Embedding from x
        Pos_emb = self.positional_embedding(x)
        Pos_emb = Pos_emb.expand(
            batch_size, seq_len, -1
        )  # Broadcast to match batch size

        # Temporal Embedding (if x_mark is provided)
        if x_mark is not None:
            Time_emb = self.temporal_embedding(x_mark)
        else:
            Time_emb = torch.zeros_like(
                Pos_emb
            )  # Fallback if no time features are provided

        # Concatenate embeddings
        X_concat = torch.cat(
            [X_token_out, Pos_emb, Time_emb], dim=-1
        )  # [batch, seq_len, total_dim]

        # Conv1D feature extraction
        X_concat = X_concat.permute(
            0, 2, 1
        )  # Switch to [batch, total_dim, seq_len] for Conv1D
        X_conv_out = self.conv1d(X_concat)  # [batch, conv_out_dim, seq_len]

        X_conv_out = X_conv_out.permute(
            0, 2, 1
        )  # Switch back to [batch, seq_len, conv_out_dim]

        return X_conv_out  # Output shape: [batch, seq_len, conv_out_dim]


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=3,
        stride=1,
        norm_type="batchnorm",
        activation_type="relu",
    ):
        super(Conv1DBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1d = nn.Conv1d(
            input_dim,  # Input dimension (features)
            output_dim,  # Output dimension (features)
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,  # To maintain the seq_len
        )
        self.norm = Normalization(norm_type, output_dim)
        self.act_func = get_activation(activation_type)

    def forward(self, x):
        # x.shape = [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # Switch to [batch, input_dim, seq_len] for Conv1D

        # Apply Conv1D
        x = self.conv1d(x)  # Resulting shape should be [batch, output_dim, seq_len]

        x = x.permute(0, 2, 1)  # Switch back to [batch, seq_len, output_dim]

        # Apply normalization
        x = self.norm(x)  # Ensure normalization is applied correctly

        # Apply activation_type
        x = self.act_func(x)

        return x  # Output shape: [batch, seq_len, output_dim]


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_d_model, output_dim, activation_type="gelu"):
        super(MLPBlock, self).__init__()

        """
        record attributes for passing the dimensions to other modules
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_d_model),
            get_activation(activation_type),
            nn.Linear(hidden_d_model, output_dim),
            get_activation(activation_type),
        )

    def forward(self, x):
        # x.shape = [batch, seq_len, input_dim]
        return self.mlp(x)  # Output shape: [batch, seq_len, output_dim]


class MHABlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, activation_type="gelu"):
        super(MHABlock, self).__init__()

        """
        record attributes for passing the dimensions to other modules
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,  # This should match the last dimension of the input tensor
            num_heads=num_heads,
            batch_first=True,  # Ensure the batch dimension is first
        )

        self.fc = nn.Linear(
            input_dim, output_dim
        )  # Adjusts to the desired output dimension
        self.act_func = get_activation(activation_type)

    def forward(self, x):
        # x.shape = [batch, seq_len, input_dim]
        attn_output, _ = self.mha(x, x, x)  # Self-attention
        attn_output = self.fc(
            attn_output
        )  # Apply a linear transformation to output_dim
        return self.act_func(attn_output)  # Output shape: [batch, seq_len, output_dim]


class Normalization(nn.Module):
    def __init__(self, norm_type, dim, affine=True, bias=True, eps=1e-5):
        super(Normalization, self).__init__()
        self.norm_type = norm_type
        self.dim = dim
        self.affine = affine
        self.bias = bias
        self.eps = eps

        if norm_type == "layernorm":
            # LayerNorm can directly handle affine transformations and bias
            self.norm = nn.LayerNorm(dim, eps=self.eps, elementwise_affine=self.affine)
        else:
            # BatchNorm1d can also handle affine transformations and bias
            self.norm = nn.BatchNorm1d(dim, eps=self.eps, affine=self.affine)

        if self.affine:
            # Initialize scale and shift parameters for additional affine transformation
            self.scale = nn.Parameter(torch.ones(dim))
            self.shift = nn.Parameter(torch.zeros(dim))
        if self.bias:
            # Learnable bias parameter
            self.learnable_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if self.norm_type == "layernorm":
            # LayerNorm expects input with last dimension as normalized_shape
            # x.shape should be [batch, seq_len, dim]
            x = self.norm(x)  # Apply LayerNorm
        else:
            # BatchNorm1d expects input in shape [batch, dim, seq_len]
            # Need to permute for BatchNorm1d
            x = x.permute(0, 2, 1)  # [batch, dim, seq_len]
            x = self.norm(x)  # Apply BatchNorm1d
            x = x.permute(0, 2, 1)  # Permute back to [batch, seq_len, dim]

        if self.affine:
            # Apply additional affine transformation: scale * x + shift
            x = x * self.scale + self.shift

        if self.bias:
            # Apply learnable bias
            x = x + self.learnable_bias

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, conv_block, feature_block, norm_type="batchnorm", activation_type="gelu"
    ):
        super(ResidualBlock, self).__init__()
        self.conv_block = conv_block
        self.feature_block = feature_block
        self.norm = Normalization(norm_type, feature_block.output_dim)
        self.activation = get_activation(activation_type)

        # Align residual only if dimensions differ
        if conv_block.input_dim != conv_block.output_dim:
            self.align_residual = nn.Linear(conv_block.input_dim, conv_block.output_dim)
        else:
            self.align_residual = None

        logger.debug(
            f"Residual block with conv_block={conv_block}, feature_block={feature_block}"
        )

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x = self.feature_block(x)

        # Align residual if necessary
        if self.align_residual is not None:
            residual = self.align_residual(residual)

        x = self.norm(x)
        x = x + residual
        return self.activation(x)


class FinalMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_d_model,
        output_dim=1,
        pred_len=1,
        activation_type="gelu",
    ):
        super(FinalMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_d_model),
            get_activation(activation_type),
            nn.Linear(hidden_d_model, output_dim * pred_len),
        )
        self.output_dim = output_dim
        self.pred_len = pred_len

    def forward(self, x):
        x = x.mean(dim=1)  # Aggregate features across the sequence
        x = self.mlp(x)
        x = x.view(
            x.size(0), self.pred_len, self.output_dim
        )  # Reshape to [batch, pred_len, output_dim]
        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # ?
        TIME_FEATURES = ["hour"]

        # Extracting configurations at the beginning
        input_dim = config.input_dim
        token_d_model = config.token_d_model
        pos_d_model = config.pos_d_model
        time_d_model = config.time_d_model
        conv_out_dim = config.conv_out_dim
        d_model = config.d_model
        hidden_d_model = config.hidden_d_model
        last_d_model = config.last_d_model
        e_layers = config.e_layers
        num_heads = config.num_heads
        fc_layer_type = config.fc_layer_type
        norm_type = config.norm_type
        activation_type = config.activation_type
        pred_len = config.pred_len
        output_dim = config.output_dim
        token_conv_kernel = config.token_conv_kernel

        if d_model % num_heads != 0:
            num_heads = max([n for n in range(1, d_model + 1) if d_model % n == 0])

        # Initialize the InitBlock
        self.init_block = InitBlock(
            input_dim=input_dim,
            token_d_model=token_d_model,
            pos_d_model=pos_d_model,
            time_d_model=time_d_model,
            conv_out_dim=conv_out_dim,
            time_features=TIME_FEATURES,
            token_conv_kernel=token_conv_kernel,
        )

        # Stack Conv1D and either MLP or MHA blocks
        layers = []
        for i in range(e_layers):
            if i == 0:
                # First layer: from conv_out_dim to d_model
                conv_block = Conv1DBlock(
                    input_dim=conv_out_dim,  # First layer takes conv_out_dim as input
                    output_dim=d_model,  # Outputs d_model
                    norm_type=norm_type,
                    activation_type=activation_type,
                )
                logger.debug(
                    f"Conv1D block [{i}], with input_dim={conv_out_dim}, output_dim={d_model}"
                )
            else:
                # Subsequent layers: from d_model to d_model
                conv_block = Conv1DBlock(
                    input_dim=d_model,  # Now input is d_model
                    output_dim=d_model,  # Output remains d_model
                    norm_type=norm_type,
                    activation_type=activation_type,
                )
                logger.debug(
                    f"Conv1D block [{i}], with input_dim={d_model}, output_dim={d_model}"
                )
            if fc_layer_type == "mha":
                feature_block = MHABlock(
                    input_dim=d_model,
                    output_dim=d_model,
                    num_heads=num_heads,
                    activation_type=activation_type,
                )
            elif fc_layer_type == "mlp":
                feature_block = MLPBlock(
                    input_dim=d_model,
                    hidden_d_model=hidden_d_model,
                    output_dim=d_model,
                    activation_type=activation_type,
                )
            else:
                raise ValueError(f"Unsupported fc_layer_type: {fc_layer_type}")

            layers.append(
                ResidualBlock(
                    conv_block,
                    feature_block,
                    norm_type=norm_type,
                    activation_type=activation_type,
                )
            )
        self.stacked_blocks = nn.Sequential(*layers)

        # Final MLP for prediction
        self.final_mlp = FinalMLP(
            input_dim=d_model,
            hidden_d_model=last_d_model,
            output_dim=output_dim,
            pred_len=pred_len,
            activation_type=activation_type,
        )

    def forward(self, x, x_mark, x_dec=None, x_dec_mark=None):
        # Ensure x has the shape [batch, seq_len, d_model]
        x = self.init_block(x, x_mark)  # Outputs in [batch, seq_len, conv_out_dim]
        x = self.stacked_blocks(x)  # Should maintain shape [batch, seq_len, d_model]
        return self.final_mlp(x)  # Output shape [batch, pred_len, output_dim]


# Define the model initialization and testing
def main():
    import argparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    test_settings = [
        # Example configuration 1: Using MLP
        {
            "input_dim": 84,  # Adjusted for divisibility and consistency
            "token_conv_kernel": 11,  # Must be larger than 3
            "seq_len": 42,  # Same as the failure setting
            "num_heads": 128,  # Same as the failure setting
            "norm_type": "layernorm",  # Same as the failure setting
            "fc_layer_type": "mlp",  # Same as the failure setting
            "token_d_model": 32,  # Same as the failure setting
            "pos_d_model": 64,  # Assuming you want to use the same value as time_d_model
            "time_d_model": 64,  # Same as the failure setting
            "conv_out_dim": 512,  # Same as the failure setting
            "d_model": 256,  # Same as the failure setting
            "hidden_d_model": 128,  # Same as the failure setting
            "last_d_model": 64,  # Same as the failure setting
            "e_layers": 2,  # Same as the failure setting
            "activation_type": "gelu",  # Assuming "gelu" as in previous examples
            "pred_len": 1,  # Assuming you want to predict a single output
            "output_dim": 1,  # Assuming the output dimension is 1
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
            activation_type=setting.get("activation_type", "gelu"),
            fc_layer_type=setting["fc_layer_type"],
            norm_type=setting["norm_type"],
            dropout=0.1,
            pred_len=setting["pred_len"],
            output_dim=setting["output_dim"],
            use_pos_enc=False,
            time_features=["hour", "quarter_hour"],
            time_d_model=setting.get(
                "time_d_model", 16
            ),  # Default to 16 if not specified
            pos_d_model=setting.get(
                "pos_d_model", 16
            ),  # Default to 16 if not specified
            token_d_model=setting.get(
                "token_d_model", 16
            ),  # Default to 16 if not specified
            conv_out_dim=setting.get(
                "conv_out_dim", 32
            ),  # Default to 32 if not specified
            d_model=setting.get("d_model", 64),  # Default to 64 if not specified
            hidden_d_model=setting.get(
                "hidden_d_model", 128
            ),  # Default to 128 if not specified
            last_d_model=setting.get(
                "last_d_model", 16
            ),  # Default to 16 if not specified
            e_layers=setting.get("e_layers", 2),  # Default to 2 if not specified
        )

        # Initialize the model with the config
        model = Model(config).to(device)

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


if __name__ == "__main__":
    main()
