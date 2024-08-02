import torch
import torch.nn as nn

from models.layers.Embed import FinalEmbedding
from models.layers.helper import Normalize, series_decomp


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # Initialize the output list with the permuted high season
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        # Iterate through the season list, excluding the last element
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            # Update out_low only if the next season exists
            if i + 2 < len(season_list):
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**i),
                        configs.seq_len // (configs.down_sampling_window**i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == "moving_avg":
            # defalut to be 25
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError("decompsition is error")

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.hidden_dim),
                nn.GELU(),
                nn.Linear(in_features=configs.hidden_dim, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=configs.hidden_dim, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            # if self.channel_independence == 0:
            #     season = self.cross_layer(season)
            #     trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super(Model, self).__init__()
        self.configs = configs
        # self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )

        self.preprocess = series_decomp(configs.moving_avg)
        self.input_dim = configs.input_dim
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = FinalEmbedding(
                1,
                configs.d_model,
            )

        self.layer = configs.e_layers
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.channel_projection_layer = nn.Linear(
                    configs.d_model, self.configs.output_dim, bias=True
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(
                        self.configs.input_dim,
                        affine=True,
                        non_norm=True if configs.use_norm == 0 else False,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        """
        self-design
        """
        self.feedforward = nn.Sequential(
            nn.Linear(self.configs.input_dim, self.configs.last_hidden_dim),
            nn.GELU(),
            nn.Linear(self.configs.last_hidden_dim, 1),
        )
        self.activation = torch.nn.ReLU()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        self-design
        """

        # Process inputs with multi-scale processing
        x_enc, x_mark_enc = self._multi_scale_process_inputs(x_enc, x_mark_enc)

        # Initialize lists to store processed inputs
        x_list = []
        x_mark_list = []

        # Process and normalize inputs
        if x_mark_enc is not None:
            for i, (x, x_mark) in enumerate(zip(x_enc, x_mark_enc)):
                Batch_Size, Seq_Len, Num_Input = x.size()
                x = self.normalize_layers[i](x, "norm")

                if self.channel_independence == 1:
                    # Combine batch and num_feature dimensions
                    x = (
                        x.permute(0, 2, 1)
                        .contiguous()
                        .reshape(Batch_Size * Num_Input, Seq_Len, 1)
                    )

                    # Repeat x_mark to match dimensions
                    x_mark = x_mark.repeat(Num_Input, 1, 1)

                x_list.append(x)
                x_mark_list.append(x_mark)

        # Pre-encoding step
        x_list = self.pre_enc(x_list)[0]

        # Embedding step
        enc_out_list = []
        if x_mark_enc is not None:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(Batch_Size, enc_out_list, x_list)

        # Stack, reshape, and average

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)

        dec_out = dec_out.view(
            Batch_Size, Num_Input, self.pred_len, self.configs.output_dim
        )
        # dec_out.shape =[batch,num_feature,pred_len,1]

        dec_out = dec_out.squeeze(-1)  # [batch, num_feature, pred_len]

        dec_out = dec_out.permute(0, 2, 1)  # [batch, pred_len, num_feature]

        # Apply the feedforward layer to combine num_features into one
        dec_out = self.feedforward(dec_out)  # [batch, pred_len, 1]

        # dec_out = self.activation(dec_out)  # [batch, pred_len, 1]

        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)

    def _multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # Select the downsampling method
        if self.configs.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.input_dim,
                out_channels=self.configs.input_dim,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc

        # Permute the input tensor to match the downsampling method
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_list = [x_enc.permute(0, 2, 1)]
        x_mark_list = [x_mark_enc]

        for _ in range(self.configs.down_sampling_layers):
            x_enc = down_pool(x_enc)
            x_enc_list.append(x_enc.permute(0, 2, 1))

            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc[:, :: self.configs.down_sampling_window, :]
                x_mark_list.append(x_mark_enc)

        return x_enc_list, x_mark_list

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []

        if self.channel_independence == 1:
            for i, (enc_out, x) in enumerate(zip(enc_out_list, x_list[0])):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # map seq_len -> pred_len
                dec_out = self.channel_projection_layer(
                    dec_out
                )  # map d_model -> output_dim
                dec_out = (
                    dec_out.reshape(
                        B * self.configs.input_dim,
                        self.configs.output_dim,
                        self.pred_len,
                    )
                    .permute(0, 2, 1)
                    .contiguous()
                )
                dec_out_list.append(dec_out)
        else:
            for i, (enc_out, x) in enumerate(zip(enc_out_list, x_list[0])):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # map seq_len -> pred_len
                out_res = (
                    x_list[1][i] if i < len(x_list[1]) else None
                )  # Get residual if available
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            # print(
            #     f"debug: x_enc.shape is {x_enc.shape}, x_mark_enc.shape is {x_mark_enc.shape}, x_dec is {x_dec}, x_mark_dec.shape is {x_mark_dec.shape}"
            # )

            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

            return dec_out_list
        else:
            raise ValueError("Only forecast tasks implemented yet")
