import os
from argparse import Namespace

import pandas as pd  # noqa
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sympy import Dict
from torch.utils.data import DataLoader

from dataset import WindPowerDataset
from utils.config import dict_to_namespace
from utils.tools import get_scaler, inverse_transform, transform  # noqa


class WindPowerDataModule(LightningDataModule):
    def __init__(self, args: Namespace | Dict):
        super().__init__()
        if isinstance(args, dict):
            args = dict_to_namespace(args, False)

        # Store the arguments
        self.args = args

        # Extract necessary attributes
        self.data_root_dir = args.data_paths.data_root_dir
        self.train_path = os.path.join(self.data_root_dir, args.data_paths.train_path)
        self.test_path = os.path.join(self.data_root_dir, args.data_paths.test_path)
        self.seq_len = args.model_settings.seq_len
        self.batch_size = args.training_settings.batch_size

        # Optional attributes with defaults
        self.train_val_split = getattr(args.data_settings, "train_val_split", 0.2)
        self.scale_x_type = getattr(args.data_settings, "scale_x_type", "standard")
        self.scale_y_type = getattr(args.data_settings, "scale_y_type", "min_max")

        # Initializing scalers
        self.scaler_x = None
        self.scaler_y = None

    def prepare_data(self):
        train_data = pd.read_csv(self.train_path)
        train_data, val_data = train_test_split(
            train_data, test_size=self.train_val_split, shuffle=False
        )
        test_data = pd.read_csv(self.test_path)

        columns = train_data.columns.tolist()
        power_index = columns.index("power")

        self.feature_columns = columns[1:power_index]  # Exclude 'time' and 'power'
        self.time_feature_columns = columns[power_index + 1 :]

        self.scaler_x = get_scaler(self.scale_x_type)
        self.scaler_y = get_scaler(self.scale_y_type)

        train_X = train_data[self.feature_columns].values
        train_y = train_data[["power"]].values

        self.scaler_x = self.scaler_x.fit(train_X) if self.scaler_x else None
        self.scaler_y = self.scaler_y.fit(train_y) if self.scaler_y else None

        self.train_X = transform(self.scaler_x, train_data[self.feature_columns].values)
        self.train_y = transform(self.scaler_y, train_data[["power"]].values)
        self.val_X = transform(self.scaler_x, val_data[self.feature_columns].values)
        self.val_y = transform(self.scaler_y, val_data[["power"]].values)
        self.test_X = transform(self.scaler_x, test_data[self.feature_columns].values)
        self.test_y = transform(self.scaler_y, test_data[["power"]].values)

        self.train_X_mark = train_data[self.time_feature_columns].values
        self.val_X_mark = val_data[self.time_feature_columns].values
        self.test_X_mark = test_data[self.time_feature_columns].values

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = WindPowerDataset(
                self.train_X, self.train_y, self.train_X_mark, self.seq_len
            )
            self.val_dataset = WindPowerDataset(
                self.val_X, self.val_y, self.val_X_mark, self.seq_len
            )
            self.test_dataset = WindPowerDataset(
                self.test_X, self.test_y, self.test_X_mark, self.seq_len
            )

        if stage in (None, "test"):
            self.test_dataset = WindPowerDataset(
                self.test_X, self.test_y, self.test_X_mark, self.seq_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=getattr(self.args, "num_workers", 4),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 4) // 2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 4) // 2,
        )

    def inverse_transform_y(self, y):
        return inverse_transform(self.scaler_y, y)


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

    # Initialize the data module
    data_module = WindPowerDataModule(args)

    # Prepare the data
    data_module.prepare_data()

    # Setup the data module (this is usually done by PyTorch Lightning, but we do it manually for testing)
    data_module.setup("fit")

    # Check the train dataloader
    train_dataloader = data_module.train_dataloader()
    print("Number of training batches:", len(train_dataloader))
    for batch in train_dataloader:
        print("Train batch shape:", batch[0].shape)
        break  # Print the shape of the first batch and exit

    # Check the validation dataloader
    val_dataloader = data_module.val_dataloader()
    print("Number of validation batches:", len(val_dataloader))
    for batch in val_dataloader:
        print("Validation batch shape:", batch[0].shape)
        break  # Print the shape of the first batch and exit

    # Setup the data module for testing
    data_module.setup("test")

    # Check the test dataloader
    test_dataloader = data_module.test_dataloader()
    print("Number of test batches:", len(test_dataloader))
    for batch in test_dataloader:
        print("Test batch shape:", batch[0].shape)
        break  # Print the shape of the first batch and exit
