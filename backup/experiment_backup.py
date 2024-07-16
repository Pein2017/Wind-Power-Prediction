# ruff: noqa: E402

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from dataset.datamodule import WindPowerDataModule
from exp.pl_exp import WindPowerExperiment
from utils.callback import MetricsCallback, get_callbacks
from utils.results import custom_test

parser = argparse.ArgumentParser(description="WindPowerPrediction")

# tb_log_dir
parser.add_argument(
    "--tb_log_dir", type=str, default="./tb_logs", help="tensorboard log dir"
)


# checkpoint_dir
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./checkpoints/",
    help="checkpoint dir for resume training",
)

parser.add_argument(
    "--train_val_split",
    type=float,
    default=0.2,
    help="validation ratio for train_val_spli",
)

parser.add_argument("--scale_x_type", type=str, default="standard", help="scale_x_type")

parser.add_argument("--scale_y_type", type=str, default="standard", help="scale_y_type")


# seed
parser.add_argument("--seed", type=int, default=17, help="random seed")

# output_dir
parser.add_argument(
    "--output_dir", type=str, default="./output", help="output dir for model"
)


parser.add_argument(
    "--model",
    type=str,
    default="SimpleMLP",
    help="Model to use for training, must be [SimpleMLP, TimeMixer]",
)

# T-Max
parser.add_argument("--T_max", type=int, default=12, help="T-Max")

# add weight_decay for Adam
parser.add_argument(
    "--weight_decay", type=float, default=1e-4, help="weight decay for Adam"
)

# time_d_model
parser.add_argument(
    "--time_d_model", type=int, default=128, help="dimension of time embedding"
)

# combine_type
parser.add_argument(
    "--combine_type",
    type=str,
    default="add",
    help="combine type for token embedding and time embedding",
)

# scheduler_type
parser.add_argument(
    "--scheduler_type", type=str, default="OneCycleLR", help="scheduler type"
)

# patience for ReduceLROnPlateau
parser.add_argument(
    "--patience", type=int, default=5, help="patience for ReduceLROnPlateau"
)


# add factor for ReduceLROnPlateau
parser.add_argument(
    "--Reduce_factor", type=float, default=0.2, help="factor for ReduceLROnPlateau"
)


parser.add_argument(
    "--min_y_value",
    type=float,
    required=False,
    default=0.0,
    help="min y value for target Power after normalization",
)


# basic config
parser.add_argument(
    "--task_name",
    type=str,
    required=False,
    default="default",
    help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
)

parser.add_argument("--is_training", type=int, default=1, help="status")


# data loader
parser.add_argument(
    "--data", type=str, required=True, default="WindPower", help="dataset type"
)


parser.add_argument(
    "--data_root_dir",
    type=str,
    default="./data/ETT/",
    help="root path of the data file",
)
parser.add_argument("--train_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument("--test_path", type=str, default="ETTh1.csv", help="data file")

parser.add_argument(
    "--target", type=str, default="power", help="target feature in S or MS task"
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)


# forecasting task
parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument(
    "--pred_len", type=int, default=1, help="prediction sequence length"
)

parser.add_argument("--inverse", type=bool, help="inverse output data", default=True)

# model define
parser.add_argument(
    "--last_hidden_dim", type=int, default=16, help="last hidden dim of FFN"
)

parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
parser.add_argument("--input_dim", type=int, default=6, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=6, help="decoder input size")
parser.add_argument("--output_dim", type=int, default=1, help="output size")
parser.add_argument("--d_model", type=int, default=16, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--hidden_dim", type=int, default=32, help="dimension of fcn")
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--decomp_method",
    type=str,
    default="moving_avg",
    help="method of series decompsition, only support moving_avg or dft_decomp",
)
parser.add_argument(
    "--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0"
)
parser.add_argument(
    "--down_sampling_layers", type=int, default=0, help="num of down sampling layers"
)
parser.add_argument(
    "--down_sampling_window", type=int, default=1, help="down sampling window size"
)
parser.add_argument(
    "--down_sampling_method",
    type=str,
    default="avg",
    help="down sampling method, only support avg, max, conv",
)


# optimization
parser.add_argument(
    "--num_workers", type=int, default=8, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch size of train input data"
)
parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=15,
    help="early stopping patience",
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="optimizer learning rate"
)

parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument("--lradj", type=str, default="TST", help="adjust learning rate")
parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)
parser.add_argument("--comment", type=str, default="none", help="com")

# gpu
parser.add_argument("--use_gpu", default="False", help="use gpu")
parser.add_argument("--gpu", type=str, default="0", help="gpu device id")
parser.add_argument("--use_multi_gpu", help="use multiple gpus", default="False")
parser.add_argument(
    "--devices", type=str, default="0", help="device ids of multiple gpus"
)


try:
    args = parser.parse_args()
except Exception as e:
    print(f"Argument error: {e}")


print("Args in experiment:")
print(args)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global args
    set_seed(args.seed)
    vared_args = vars(args)  # Convert Namespace to dictionary

    args.use_multi_gpu = args.use_multi_gpu == "True"
    args.use_gpu = args.use_gpu == "True"

    # Determine device settings based on use_multi_gpu flag
    if args.use_multi_gpu:
        devices = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        devices = [int(args.gpu)]
        strategy = "auto"

    # Initialize TensorBoard logger
    logger = TensorBoardLogger(save_dir=args.tb_log_dir, name="wind_power")

    # Prepare data module
    data_module = WindPowerDataModule(vared_args)
    data_module.prepare_data()
    data_module.setup("fit")

    # Calculate steps_per_epoch from train_dataloader
    train_dataloader = data_module.train_dataloader()
    steps_per_epoch = len(train_dataloader)
    args.steps_per_epoch = steps_per_epoch

    # Initialize model
    model = WindPowerExperiment(args)

    # Get callbacks
    callbacks = get_callbacks(args)
    callbacks.append(
        MetricsCallback(
            criterion=model.criterion,
        )
    )

    # Initialize trainer with DDP strategy for multi-GPU training if use_multi_gpu is True
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Custom test the model and visualize results
    custom_test(model, data_module, exp_settings=args.setting)


if __name__ == "__main__":
    main()
