import os
from argparse import Namespace
from typing import Union

import pandas as pd  # noqa
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sympy import Dict
from torch.utils.data import DataLoader

from dataset import WindPowerDataset
from utils.config import dict_to_namespace
from utils.tools import get_skl_scaler  # noqa


class WindPowerDataModule(LightningDataModule):
    def __init__(self, args: Union[Namespace, Dict]):
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
        self.val_split = getattr(args.data_settings, "val_split ", 0.2)
        self.random_split = getattr(args.data_settings, "random_split")

        # check if val_split  is smaller than 0.4
        if self.val_split > 0.4:
            raise ValueError("val_split  should be smaller than 0.4")

        self.scale_x_type = getattr(args.data_settings, "scale_x_type", "standard")
        self.scale_y_type = getattr(args.data_settings, "scale_y_type", "min_max")

        # Transform order for y
        self.y_transform_order = getattr(args.data_settings, "y_transform_order", None)

        # Initialize the YPowerScaler
        self.scaler_x = YPowerScaler(get_skl_scaler(self.scale_x_type))
        self.scaler_y = YPowerScaler(
            get_skl_scaler(self.scale_y_type), self.y_transform_order
        )

    def prepare_data(self):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)

        if self.random_split:
            train_data, val_data = train_test_split(
                train_data,
                test_size=self.val_split,
            )
        else:
            train_data_size = int(len(train_data) * (1 - self.val_split))
            train_data, val_data = (
                train_data[:train_data_size],
                train_data[train_data_size:],
            )

        columns = train_data.columns.tolist()
        power_index = columns.index("power")

        self.feature_columns = columns[1:power_index]  # Exclude 'time' and 'power'
        self.time_feature_columns = columns[power_index + 1 :]

        train_X = train_data[self.feature_columns].values
        train_y = train_data[["power"]].values

        self.scaler_x.fit(train_X)
        self.scaler_y.fit(train_y)

        self.train_X = self.scaler_x.transform(train_X)
        self.train_y = self.scaler_y.transform(train_y)
        self.val_X = self.scaler_x.transform(val_data[self.feature_columns].values)
        self.val_y = self.scaler_y.transform(val_data[["power"]].values)
        self.test_X = self.scaler_x.transform(test_data[self.feature_columns].values)
        self.test_y = self.scaler_y.transform(test_data[["power"]].values)

        self.train_X_mark = train_data[self.time_feature_columns].values
        self.val_X_mark = val_data[self.time_feature_columns].values
        self.test_X_mark = test_data[self.time_feature_columns].values
        print("*" * 30)
        print("Data preparation completed!")
        if self.y_transform_order:
            print(f"Transformed y with order {self.y_transform_order}")

        print("Data shapes:")
        print("Train X shape:", self.train_X.shape)
        print("Train y shape:", self.train_y.shape)
        print("Val X shape:", self.val_X.shape)
        print("Val y shape:", self.val_y.shape)
        print("Test X shape:", self.test_X.shape)
        print("Test y shape:", self.test_y.shape)
        print(f"Train X mark shape: {self.train_X_mark.shape}")
        print(f"Val X mark shape: {self.val_X_mark.shape}")
        print(f"Test X mark shape: {self.test_X_mark.shape}")

        print("*" * 30)

    def transform(self, scaler, data, order=None):
        if order is not None:
            data = data**order  # Apply the power transformation
        if scaler:
            data = scaler.transform(data)  # Apply the scaling
        return data

    def inverse_transform(self, scaler, data, order=None):
        if scaler:
            data = scaler.inverse_transform(data)  # Inverse the scaling
        if order is not None:
            data = data ** (1 / order)  # Reverse the power transformation
        return data

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
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=getattr(self.args, "num_workers", 4),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 4) // 2,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 4) // 2,
        )


class YPowerScaler:
    """
    Transform the order of Y first and then apply sklearn scaler
    """

    def __init__(self, scaler=None, y_transform_order=None):
        self.scaler = scaler
        self.y_transform_order = y_transform_order

    def fit(self, data):
        """Fit the scaler to the data after applying the power transformation."""
        if self.y_transform_order is not None:
            data = data**self.y_transform_order  # Apply the power transformation
        if self.scaler:
            self.scaler.fit(data)  # Fit the scaling
        return self

    def transform(self, data):
        """Transform the data using the fitted scaler."""
        if self.y_transform_order is not None:
            data = data**self.y_transform_order  # Apply the power transformation
        if self.scaler:
            data = self.scaler.transform(data)  # Apply the scaling
        return data

    def inverse_transform(self, data):
        """Inverse transform the data back to the original scale."""
        if self.scaler:
            data = self.scaler.inverse_transform(data)  # Inverse the scaling
        if self.y_transform_order is not None:
            data = data ** (
                1 / self.y_transform_order
            )  # Reverse the power transformation
        return data


if __name__ == "__main__":
    seq_len = 5  # example sequence length
    batch_size = 32  # example batch size

    data_dir = "/data/Pein/Pytorch/Wind-Power-Prediction/data"
    train_ori = os.path.join(data_dir, "train_data_89_withTime.csv")
    test_ori = os.path.join(data_dir, "test_data_89_withTime.csv")

    # Create the args dictionary
    args = {
        "data_root_dir": data_dir,
        "train_path": train_ori,
        "test_path": test_ori,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "val_split ": 0.2,
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
