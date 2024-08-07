import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the default log level to DEBUG to capture all levels of logs
logger.setLevel(logging.INFO)

# Create a console handler and set the log level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler and set the log level to DEBUG
log_file = "embed_debug.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both the handlers
formatter = logging.Formatter("%(asctime)s %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, token_emb_kernel_size=3):
        super(TokenEmbedding, self).__init__()
        self.token_emb_kernel_size = token_emb_kernel_size
        self.tokenConv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=token_emb_kernel_size,
            padding=0,  # Set padding to 0, we will handle it manually
            padding_mode="zeros",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # Calculate the padding size
        padding_left = (self.token_emb_kernel_size - 1) // 2
        padding_right = self.token_emb_kernel_size // 2

        # Apply padding
        x = F.pad(
            x.permute(0, 2, 1), (padding_left, padding_right), mode="constant", value=0
        )
        x = self.tokenConv(x).transpose(1, 2)
        return x


"""
when time featues are 6 for pwoer prediction. 6 = (Hour and quater_hour) * 3
"""
# class TemporalFeatureEmbedding(nn.Module):
#     def __init__(self, time_d_model, combine_type="add"):
#         super(TemporalFeatureEmbedding, self).__init__()
#         self.d_model = time_d_model
#         self.combine_type = combine_type

#         # Embeddings for discrete time features
#         self.hour_embed = nn.Embedding(24, time_d_model)
#         self.quarter_hour_embed = nn.Embedding(4, time_d_model)

#         # Linear layers for sine and cosine transformations
#         self.linear_hour_sin = nn.Linear(1, time_d_model)
#         self.linear_hour_cos = nn.Linear(1, time_d_model)
#         self.linear_quarter_hour_sin = nn.Linear(1, time_d_model)
#         self.linear_quarter_hour_cos = nn.Linear(1, time_d_model)

#         self.dropout = nn.Dropout(0.1)  # Dropout rate (optional)

#     def forward(self, x):
#         # x: [batch, Seq_Len, 6] - containing hour, quarter_hour, hour_sin, hour_cos, quarter_hour_sin, quarter_hour_cos

#         # Extract features
#         hour = x[:, :, 0].long()
#         quarter_hour = x[:, :, 1].long()
#         hour_sin = x[:, :, 2].unsqueeze(-1)
#         hour_cos = x[:, :, 3].unsqueeze(-1)
#         quarter_hour_sin = x[:, :, 4].unsqueeze(-1)
#         quarter_hour_cos = x[:, :, 5].unsqueeze(-1)

#         # Apply embeddings and linear layers
#         hour_embedding = self.hour_embed(hour)
#         quarter_hour_embedding = self.quarter_hour_embed(quarter_hour)
#         hour_sin_embedding = self.linear_hour_sin(hour_sin)
#         hour_cos_embedding = self.linear_hour_cos(hour_cos)
#         quarter_hour_sin_embedding = self.linear_quarter_hour_sin(quarter_hour_sin)
#         quarter_hour_cos_embedding = self.linear_quarter_hour_cos(quarter_hour_cos)

#         # Combine embeddings
#         if self.combine_type == "concat":
#             combined_embedding = torch.cat(
#                 [
#                     hour_embedding,
#                     quarter_hour_embedding,
#                     hour_sin_embedding,
#                     hour_cos_embedding,
#                     quarter_hour_sin_embedding,
#                     quarter_hour_cos_embedding,
#                 ],
#                 dim=-1,
#             )
#         else:
#             combined_embedding = (
#                 hour_embedding
#                 + quarter_hour_embedding
#                 + hour_sin_embedding
#                 + hour_cos_embedding
#                 + quarter_hour_sin_embedding
#                 + quarter_hour_cos_embedding
#             )

#         # Apply dropout (optional)
#         combined_embedding = self.dropout(combined_embedding)

#         return combined_embedding


# class FinalEmbedding(nn.Module):
#     def __init__(self, input_dim, token_d_model, time_d_model=-1, combine_type="add"):
#         super(FinalEmbedding, self).__init__()
#         if time_d_model == -1:
#             time_d_model = token_d_model
#             print(
#                 f"time_d_model is not set, defaulting to token_d_model: {time_d_model}"
#             )

#         self.token_embedding = TokenEmbedding(input_dim, token_d_model)
#         self.temporal_embedding = TemporalFeatureEmbedding(time_d_model, combine_type)
#         self.combine_type = combine_type
#         self.token_d_model = token_d_model
#         self.time_d_model = time_d_model

#         if combine_type == "add" and token_d_model != time_d_model:
#             self.temporal_to_token_dim = nn.Linear(time_d_model, token_d_model)

#     def forward(self, x, x_mark):
#         # x: [batch, Seq_len, input_dim]
#         token_emb = self.token_embedding(
#             x
#         )  # token embedding -> [batch, Seq_len, token_d_model]

#         temporal_emb = self.temporal_embedding(x_mark)
#         # add :   [batch, Seq_len, time_d_model]
#         # concat: [batch, Seq_len, 6 * time_d_model]

#         # Combine embeddings
#         if self.combine_type == "concat":
#             combined_emb = torch.cat([token_emb, temporal_emb], dim=-1)
#             # combined_emb: [batch, Seq_len, token_d_model + 6 * time_d_model]
#         else:
#             if self.time_d_model != self.token_d_model:
#                 temporal_emb = self.temporal_to_token_dim(temporal_emb)
#             combined_emb = token_emb + temporal_emb
#             # combined_emb: [batch, Seq_len, token_d_model]

#         return combined_emb

"""
when time featues are n_time_feature for power prediction.
"""


class TemporalFeatureEmbedding(nn.Module):
    def __init__(
        self,
        time_d_model,
        combine_type="add",
        max_len=5000,
        use_pos_enc=True,
        time_features=[
            "hour",
            "quarter_hour",
        ],  # "day"
    ):
        super(TemporalFeatureEmbedding, self).__init__()
        self.d_model = time_d_model
        self.combine_type = combine_type
        self.use_pos_enc = use_pos_enc
        self.time_features = time_features

        # Embeddings for discrete time features
        self.hour_embed = nn.Embedding(24, time_d_model)
        self.quarter_hour_embed = nn.Embedding(4, time_d_model)
        self.day_embed = nn.Embedding(
            32, time_d_model
        )  # Assuming up to 31 days in a month
        self.day_in_week_embed = nn.Embedding(7, time_d_model)

        # Linear layers for sine and cosine transformations
        self.linear_sin = nn.Linear(1, time_d_model)
        self.linear_cos = nn.Linear(1, time_d_model)

        self.dropout = nn.Dropout(0.1)  # Dropout rate (optional)

        # Positional encoding
        if self.use_pos_enc:
            self.positional_embedding = PositionalEmbedding(time_d_model, max_len)

    def forward(self, x):
        embeddings = []

        # Embedding and linear layers application based on selected time features
        feature_idx = {"hour": 0, "quarter_hour": 1, "day": 2, "day_in_week": 3}

        for feature in self.time_features:
            idx = feature_idx[feature]
            discrete_feature = x[:, :, idx].long()
            sin_feature = x[:, :, idx + 4].unsqueeze(-1)
            cos_feature = x[:, :, idx + 5].unsqueeze(-1)

            if feature == "hour":
                embeddings.append(self.hour_embed(discrete_feature))
            elif feature == "quarter_hour":
                embeddings.append(self.quarter_hour_embed(discrete_feature))
            elif feature == "day":
                embeddings.append(self.day_embed(discrete_feature))
            elif feature == "day_in_week":
                embeddings.append(self.day_in_week_embed(discrete_feature))

            embeddings.append(self.linear_sin(sin_feature))
            embeddings.append(self.linear_cos(cos_feature))

        # Combine embeddings
        if self.combine_type == "concat":
            combined_embedding = torch.cat(embeddings, dim=-1)
        else:
            combined_embedding = sum(embeddings)

        # Apply positional encoding if enabled
        if self.use_pos_enc:
            positional_embedding = self.positional_embedding(x)
            if self.combine_type == "concat":
                positional_embedding = positional_embedding.expand(x.size(0), -1, -1)
                combined_embedding = torch.cat(
                    [combined_embedding, positional_embedding], dim=-1
                )
            else:
                # Ensure positional_embedding and combined_embedding have the same dimensions
                if positional_embedding.size(-1) != combined_embedding.size(-1):
                    positional_embedding = positional_embedding.expand_as(
                        combined_embedding
                    )
                combined_embedding += positional_embedding

        # Apply dropout (optional)
        combined_embedding = self.dropout(combined_embedding)

        logger.debug(
            f"debug: temporal.shape of combined_embedding is {combined_embedding.shape}"
        )

        return combined_embedding


class FinalEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        token_d_model,
        time_d_model=-1,
        combine_type="add",
        token_emb_kernel_size=3,
        time_features=[
            "hour",
            "quarter_hour",
        ],  # "day"
    ):
        super(FinalEmbedding, self).__init__()
        if time_d_model == -1:
            time_d_model = token_d_model
            logger.debug(
                f"time_d_model is not set, defaulting to token_d_model: {time_d_model}"
            )

        self.token_embedding = TokenEmbedding(
            input_dim,
            token_d_model,
            token_emb_kernel_size=token_emb_kernel_size,
        )
        self.temporal_embedding = TemporalFeatureEmbedding(
            time_d_model, combine_type=combine_type, time_features=time_features
        )
        self.combine_type = combine_type
        self.token_d_model = token_d_model
        self.time_d_model = time_d_model

        if combine_type == "add" and token_d_model != time_d_model:
            self.temporal_to_token_dim = nn.Linear(time_d_model, token_d_model)

    def forward(self, x, x_mark):
        # x: [batch, Seq_len, input_dim]
        token_emb = self.token_embedding(x)
        logger.debug(f"shape of token_emb: {token_emb.shape}")  # Debugging
        # token_emb: [batch, Seq_len, token_d_model]

        temporal_emb = self.temporal_embedding(x_mark)
        logger.debug(f"shape of temporal_emb: {temporal_emb.shape}")  # Debugging
        # temporal_emb: [batch, Seq_len, time_d_model]

        # Combine embeddings
        if self.combine_type == "concat":
            combined_emb = torch.cat([token_emb, temporal_emb], dim=-1)
            logger.debug(f"shape of combined_emb: {combined_emb.shape}")  # Debugging
            # combined_emb: [batch, Seq_len, token_d_model + time_d_model]
        else:
            if self.token_d_model != self.time_d_model:
                temporal_emb = self.temporal_to_token_dim(temporal_emb)
            combined_emb = token_emb + temporal_emb
            # combined_emb: [batch, Seq_len, token_d_model]

        return combined_emb


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(input_dim=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


# 生成时间序列数据
def generate_time_series(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq="15T")
    df_stamp = pd.DataFrame({"date": dates})
    df_stamp["month"] = df_stamp["date"].dt.month
    df_stamp["day"] = df_stamp["date"].dt.day
    df_stamp["weekday"] = df_stamp["date"].dt.weekday
    df_stamp["hour"] = df_stamp["date"].dt.hour
    df_stamp["minute"] = df_stamp["date"].dt.minute
    df_stamp["hour_sin"] = np.sin(2 * np.pi * df_stamp["hour"] / 24)
    df_stamp["hour_cos"] = np.cos(2 * np.pi * df_stamp["hour"] / 24)
    df_stamp["day_sin"] = np.sin(2 * np.pi * df_stamp["day"] / 31)
    df_stamp["day_cos"] = np.cos(2 * np.pi * df_stamp["day"] / 31)
    df_stamp["weekday_sin"] = np.sin(2 * np.pi * df_stamp["weekday"] / 7)
    df_stamp["weekday_cos"] = np.cos(2 * np.pi * df_stamp["weekday"] / 7)
    df_stamp["month_sin"] = np.sin(2 * np.pi * df_stamp["month"] / 12)
    df_stamp["month_cos"] = np.cos(2 * np.pi * df_stamp["month"] / 12)
    x_mark = df_stamp[
        [
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
        ]
    ].values
    return x_mark


def visualize_temporal_embedding(model, x_mark, filename="temporal_embedding.png"):
    with torch.no_grad():
        embeddings = model(x_mark)
    embeddings = embeddings.squeeze().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(embeddings)
    plt.title("Temporal Embeddings Visualization")
    plt.xlabel("Time Step")
    plt.ylabel("Embedding Value")
    plt.legend(["Hour", "Weekday", "Day", "Month", "Minute"])
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def visualize_solar_embedding(model, time_index, filename="solar_embedding.png"):
    with torch.no_grad():
        embeddings = model(time_index)
    embeddings = embeddings.squeeze().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(embeddings)
    plt.title("Solar Irradiance Embeddings Visualization")
    plt.xlabel("Time Step")
    plt.ylabel("Embedding Value")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    batch_size, seq_len, c_in = 32, 10, 3  # Example dimensions
    d_model = 2  # Dimension of the model

    # Example input tensors
    x = torch.randn(batch_size, seq_len, c_in)  # Input features
    time_index = torch.randint(0, 96, (batch_size, seq_len))  # Time indices

    # Create and test the final embedding layer
    final_embedding_add = FinalEmbedding(c_in, d_model, combine_type="add")
    final_embedding_concat = FinalEmbedding(c_in, d_model, combine_type="concat")

    output_add = final_embedding_add(x, time_index)
    output_concat = final_embedding_concat(x, time_index)

    print(
        "Output shape with 'add':", output_add.shape
    )  # Should output torch.Size([32, 10, 64])
    print(
        "Output shape with 'concat':", output_concat.shape
    )  # Should output torch.Size([32, 10, 64])
