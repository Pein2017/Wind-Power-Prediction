import itertools


def generate_hyperparameter_combinations(base_config, search_space):
    """Generate combinations of hyperparameters based on the search space."""
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        config = base_config.copy()
        config.update(params)
        yield config
