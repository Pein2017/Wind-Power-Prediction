import argparse
import datetime

import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WindPowerPrediction")
    parser.add_argument(
        "--config",
        type=str,
        default="/data3/lsf/Pein/Power-Prediction/config/optuna_config.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--exp_time_str",
        type=str,
        default=datetime.datetime.now().strftime("%y-%m-%d"),
        help="Experiment time string for path substitution",
    )
    return parser.parse_args()


def load_config(config_path, time_str):
    """Load the configuration from a YAML file and substitute paths with the time string."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Substitute paths with the time string
    for key, value in config.items():
        if isinstance(value, str) and "exp_time_str" in value:
            config[key] = value.replace("exp_time_str", time_str)

    return config


def create_exp_settings(args):
    """Create experiment setting string if not provided."""
    if not hasattr(args, "exp_settings") or not args.exp_settings:
        args.exp_settings = (
            f"seq_len-{args.seq_len}-"
            f"lr-{args.learning_rate}-"
            f"d-{args.d_model}-"
            f"hid_d-{args.hidden_dim}-"
            f"last_d-{args.last_hidden_dim}-"
            f"time_d-{args.time_d_model}-"
            f"e_layers-{args.e_layers}-"
            f"token_emb_kernel_size-{args.token_emb_kernel_size}-"
            f"dropout-{args.dropout}-"
            f"comb_type-{args.combine_type}-"
            f"bs-{args.batch_size}"
        )
    return args


def dict_to_namespace(d, if_create_exp_settings=True):
    """Convert a dictionary to an argparse.Namespace and complete exp_settings if necessary."""
    args = argparse.Namespace(**d)
    if if_create_exp_settings:
        args = create_exp_settings(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.exp_time_str)
    args = dict_to_namespace(config)
    print(args)
