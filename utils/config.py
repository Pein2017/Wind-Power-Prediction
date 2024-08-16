import argparse
import datetime
from argparse import Namespace

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

    # Function to recursively substitute placeholders
    def substitute_placeholders(config):
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and "exp_time_str" in value:
                    config[key] = value.replace("exp_time_str", time_str)
                elif isinstance(value, (dict, list)):
                    config[key] = substitute_placeholders(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                config[i] = substitute_placeholders(item)
        return config

    config = substitute_placeholders(config)

    # Convert the final dictionary to Namespace
    return dict_to_namespace(config, False)


def create_exp_settings(args):
    """Create experiment setting string if not provided."""
    if not hasattr(args, "exp_settings") or not args.exp_settings:
        # Ensure model_settings and training_settings are Namespace objects
        model_settings = args.model_settings
        training_settings = args.training_settings

        # Round the learning rate to 4 decimal places
        learning_rate_rounded = round(training_settings.learning_rate, 4)

        # Build the exp_settings string based on nested attributes
        args.exp_settings = (
            f"seq_len-{model_settings.seq_len}-"
            f"lr-{learning_rate_rounded}-"
            f"d-{model_settings.d_model}-"
            f"hid_d-{model_settings.hidden_d_model}-"
            f"last_d-{model_settings.last_d_model}-"
            f"tok_d-{model_settings.token_d_model}-"
            f"time_d-{model_settings.time_d_model}-"
            f"pos_d-{model_settings.pos_d_model}-"
            f"e_layers-{model_settings.e_layers}-"
            f"tok_conv_k-{model_settings.token_conv_kernel}-"
            # f"dropout-{model_settings.dropout}-"
            f"bs-{training_settings.batch_size}-"
            f"norm_type-{model_settings.norm_type}"
            # f"num_heads-{model_settings.num_heads}-"
        )

    return args


def namespace_to_dict(namespace):
    """Convert a Namespace to a dictionary."""
    if isinstance(namespace, Namespace):
        return {key: namespace_to_dict(value) for key, value in vars(namespace).items()}
    elif isinstance(namespace, list):
        return [namespace_to_dict(item) for item in namespace]
    else:
        return namespace


def dict_to_namespace(dictionary, if_create_exp_settings=False):
    """Convert a dictionary to a Namespace and complete exp_settings if necessary."""

    def recursive_dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = recursive_dict_to_namespace(value)
                elif isinstance(value, list):
                    d[key] = [
                        recursive_dict_to_namespace(v) if isinstance(v, dict) else v
                        for v in value
                    ]
            return Namespace(**d)
        return d

    args = recursive_dict_to_namespace(dictionary)
    if if_create_exp_settings:
        args = create_exp_settings(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.exp_time_str)
    # args = dict_to_namespace(config)
    print(config)
