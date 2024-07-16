import argparse

import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WindPowerPrediction")
    parser.add_argument(
        "--config",
        type=str,
        default="/data3/lsf/Pein/Power-Prediction/config/base_config.yaml",
        help="Path to base configuration file",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_exp_settings(args):
    """Create experiment setting string if not provided."""
    if not args.exp_settings:
        args.exp_settings = (
            f"seq_len-{args.seq_len}-"
            f"lr-{args.learning_rate}-"
            f"d-{args.d_model}-"
            f"hid_d-{args.hidden_dim}-"
            f"last_d-{args.last_hidden_dim}-"
            f"time_d-{args.time_d_model}-"
            f"e_layers-{args.e_layers}-"
            f"comb_type-{args.combine_type}-"
            f"bs-{args.batch_size}"
        )
    return args


def dict_to_namespace(d):
    """Convert a dictionary to an argparse.Namespace and complete exp_settings if necessary."""
    args = argparse.Namespace(**d)
    return create_exp_settings(args)
