import itertools
import sys

sys.path.append("/data3/lsf/Pein/Power-Prediction")


from utils.config import dict_to_namespace


def generate_hyperparameter_combinations(base_config, search_space):
    """Generate combinations of hyperparameters based on the search space."""
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))

        # Create a temporary dictionary from the base config
        config = base_config.__dict__.copy()
        config.update(params)

        # Convert the updated dictionary back to Namespace
        config_namespace = dict_to_namespace(config, False)

        yield config_namespace


if __name__ == "__main__":
    from run_scripts.run_exp import SEARCH_SPACE
    from utils.config import dict_to_namespace, load_config, parse_args  # noqa

    # Example usage:
    args = parse_args()
    base_config = load_config(args.config, args.exp_time_str)

    # Generate hyperparameter combinations
    configurations = list(
        generate_hyperparameter_combinations(base_config, SEARCH_SPACE)
    )
    # configurations = [dict_to_namespace(config) for config in configurations]

    print(f"Total number of experiments: {len(configurations)}")
    print(f"First experiment: {configurations[0]}")
