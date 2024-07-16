import torch
import torch.nn as nn

from models.layers.Embed import FinalEmbedding


class MLPBlock(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, activation=nn.GELU(), dropout=0.1
    ):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.skip_connection = nn.Linear(input_dim, output_dim)
        self.bn_skip = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = self.skip_connection(x)
        identity = self.bn_skip(identity.transpose(1, 2)).transpose(1, 2)

        out = self.fc1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out += identity
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
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
        self.min_y_value = configs.min_y_value
        print(f"\nmin_y_value has been set to {self.min_y_value}\n")
        self.min_y_value = torch.tensor(
            configs.min_y_value, dtype=torch.float32, device=torch.device("cuda:0")
        )

        self.initial_embedding = FinalEmbedding(
            input_dim, token_d_model, time_d_model, combine_type=combine_type
        )

        # Adjust the input dimension for the first MLPBlock based on combine_type
        if combine_type == "concat":
            first_block_input_dim = token_d_model + 6 * time_d_model
        else:
            # add mode
            first_block_input_dim = token_d_model
        self.normalization = nn.BatchNorm1d(first_block_input_dim)

        self.mlp_blocks = nn.ModuleList()
        self.mlp_blocks.append(
            MLPBlock(first_block_input_dim, hidden_dim, hidden_dim, dropout=dropout)
        )
        for _ in range(e_layers - 2):
            self.mlp_blocks.append(
                MLPBlock(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            )
        self.mlp_blocks.append(
            MLPBlock(hidden_dim, hidden_dim, last_hidden_dim, dropout=dropout)
        )

        self.final_fc = nn.Linear(last_hidden_dim, output_dim)
        self.prediction_fc = nn.Linear(
            self.seq_len * output_dim, output_dim * self.pred_len
        )

    def forward(self, x, x_mark, x_dec, x_dec_mark, mode="norm"):
        x = self.initial_embedding(x, x_mark)
        x = self.normalization(x.transpose(1, 2)).transpose(1, 2)
        for block in self.mlp_blocks:
            x = block(x)
        x = self.final_fc(x)

        # Flatten x to [batch_size, -1] before the final prediction layer
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(batch_size, -1)
        x = self.prediction_fc(x)

        # Reshape to [batch_size, pred_len, output_dim]
        out = x.view(batch_size, self.pred_len, self.output_dim)

        return out

    # for Solar Power Prediction
    # def forward(self, x, x_mark, x_dec, x_dec_mark, mode="norm"):
    #     x = self.initial_embedding(x, x_mark)
    #     x = self.normalization(x.transpose(1, 2)).transpose(1, 2)
    #     for block in self.mlp_blocks:
    #         x = block(x)
    #     x = self.final_fc(x)

    #     # Flatten x to [batch_size, -1] before the final prediction layer
    #     batch_size, seq_len, hidden_dim = x.shape
    #     x = x.view(batch_size, -1)
    #     x = self.prediction_fc(x)

    #     # Reshape to [batch_size, pred_len, output_dim]
    #     out = x.view(batch_size, self.pred_len, self.output_dim)

    #     # Apply prior knowledge to mask the output values during night hours
    #     night_hours_mask = (x_dec_mark[:, -1, 0] >= 80) | (x_dec_mark[:, -1, 0] <= 20)
    #     night_hours_mask = (
    #         night_hours_mask.unsqueeze(-1)
    #         .unsqueeze(-1)
    #         .expand(batch_size, self.pred_len, self.output_dim)
    #     )
    #     out = torch.where(night_hours_mask, self.min_y_value, out)

    #     return out
