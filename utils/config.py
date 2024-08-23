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
        default="/data/Pein/Pytorch/Wind-Power-Prediction/config/base_config.yaml",
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
        data_settings = args.data_settings
        scheduler_settings = args.scheduler_settings

        norm_after_dict = {
            "conv_norm": model_settings.conv_norm,
            "mlp_norm": model_settings.mlp_norm,
        }

        # Round the learning rate to 4 decimal places
        learning_rate_rounded = round(training_settings.learning_rate, 4)
        dropout_rounded = round(model_settings.dropout, 3)
        weight_decay_rounded = round(scheduler_settings.weight_decay, 4)
        y_transform_order_rounded = round(data_settings.y_transform_order, 3)

        # Define the settings dictionary
        settings_dict = {
            # Data Settings
            "v_splt": data_settings.val_split,  # val_split -> v_splt
            "y_ord": y_transform_order_rounded,  # y_transform_order -> y_ord
            # Model Architecture Settings
            "d": model_settings.d_model,  # d_model -> d
            "hd": model_settings.hidden_d_model,  # hidden_d_model -> hd
            "ld": model_settings.last_d_model,  # last_d_model -> ld
            "tok_d": model_settings.token_d_model,  # token_d_model -> tok_d
            "time_d": model_settings.time_d_model,  # time_d_model -> time_d
            "pos_d": model_settings.pos_d_model,  # pos_d_model -> pos_d
            "lyrs": model_settings.e_layers,  # e_layers -> lyrs
            "n_heads": model_settings.num_heads,  # num_heads -> n_heads
            "conv_d": model_settings.conv_out_dim,  # conv_out_dim -> conv_d
            "tok_kern": model_settings.token_conv_kernel,  # token_conv_kernel -> tok_kern
            "feat_kern": model_settings.feat_conv_kernel,  # feat_conv_kernel -> feat_kern
            "norm_typ": model_settings.norm_type,  # norm_type -> norm_typ
            "skp_mod": model_settings.skip_connection_mode,  # skip_connection_mode -> skp_mod
            "use_pos_enc": model_settings.use_pos_enc,  # use_pos_enc -> use_pos_enc
            "seq_len": model_settings.seq_len,  # seq_len -> seq_len
            "mlp_norm": norm_after_dict["mlp_norm"],  # mlp_norm -> mlp_norm
            # Training Settings
            "ep": training_settings.train_epochs,  # train_epochs -> ep
            "bs": training_settings.batch_size,  # batch_size -> bs
            "lr": learning_rate_rounded,  # learning_rate -> lr
            "dp": dropout_rounded,  # dropout -> dp
            "w_decay": weight_decay_rounded,  # weight_decay -> w_decay
        }

        # Create the exp_settings string
        args.exp_settings = "-".join(
            [f"{key}-{value}" for key, value in settings_dict.items()]
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
