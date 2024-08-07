import torch
import torch.nn as nn

from models.layers.Embed import FinalEmbedding


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation=nn.GELU(),
        dropout=0.1,
    ):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_connection = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=stride
        )
        self.bn_skip = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = self.skip_connection(x)
        identity = self.bn_skip(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_dim = configs.output_dim

        input_dim = configs.input_dim
        token_d_model = configs.d_model
        time_d_model = configs.time_d_model
        hidden_dim = configs.hidden_dim
        combine_type = configs.combine_type
        last_hidden_dim = configs.last_hidden_dim
        output_dim = configs.output_dim
        e_layers = configs.e_layers
        dropout = configs.dropout
        token_emb_kernel_size = configs.token_emb_kernel_size
        self.min_y_value = configs.min_y_value

        self.min_y_value = torch.tensor(
            configs.min_y_value, dtype=torch.float32, device=torch.device("cuda:0")
        )

        self.initial_embedding = FinalEmbedding(
            input_dim,
            token_d_model,
            time_d_model,
            combine_type=combine_type,
            token_emb_kernel_size=token_emb_kernel_size,
        )

        if combine_type == "concat":
            first_block_input_dim = token_d_model + 6 * time_d_model
        else:
            first_block_input_dim = token_d_model
        self.normalization = nn.BatchNorm1d(first_block_input_dim)

        self.conv_blocks = nn.ModuleList()
        in_channels = first_block_input_dim
        out_channels = hidden_dim

        for i in range(e_layers - 1):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels, out_channels, kernel_size=3, stride=2, dropout=dropout
                )
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, last_hidden_dim)

        self.final_conv = ConvBlock(
            in_channels, last_hidden_dim, kernel_size=3, stride=1, dropout=dropout
        )
        self.final_fc = nn.Linear(last_hidden_dim, output_dim)
        conv_seq_len = self.seq_len // (2 ** (e_layers - 1))
        self.prediction_fc = nn.Linear(
            conv_seq_len * output_dim, output_dim * self.pred_len
        )

    def forward(self, x, x_mark, x_dec, x_dec_mark, mode="norm"):
        x = self.initial_embedding(x, x_mark)
        x = self.normalization(x.transpose(1, 2)).transpose(1, 2)

        x = x.transpose(1, 2)  # Convert to [batch_size, channels, seq_len]
        for block in self.conv_blocks:
            x = block(x)
        x = self.final_conv(x)

        x = x.transpose(1, 2)  # Convert back to [batch_size, seq_len, channels]
        x = self.final_fc(x)

        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(batch_size, -1)
        x = self.prediction_fc(x)
        out = x.view(batch_size, self.pred_len, self.output_dim)

        return out
