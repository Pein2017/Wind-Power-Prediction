"""
Temporally not used.
"""

import os

from dataset.datamodule import WindPowerDataModule  # noqa


def prepare_data(args):
    data_module = WindPowerDataModule(args)
    data_module.prepare_data()
    return data_module


# Example usage
if __name__ == "__main__":
    seq_len = 5  # example sequence length
    batch_size = 32  # example batch size

    data_dir = "/data3/lsf/Pein/Power-Prediction/data"
    train_ori = os.path.join(data_dir, "train_data_89_withTime.csv")
    test_ori = os.path.join(data_dir, "test_data_89_withTime.csv")

    # Create the args dictionary
    args = {
        "data_root_dir": data_dir,
        "train_path": train_ori,
        "test_path": test_ori,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "train_val_split": 0.2,
        "scale_x_type": "standard",
        "scale_y_type": "min_max",
        "num_workers": 4,
    }

    data_module = prepare_data(args)
    data_module.setup("fit")

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    print("Train DataLoader:", len(train_dataloader))
    print("Validation DataLoader:", len(val_dataloader))

    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()
    print("Test DataLoader:", len(test_dataloader))
