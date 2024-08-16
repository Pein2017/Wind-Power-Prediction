import argparse
import sys
import unittest

sys.path.append("models")

import torch
from layers.Embed import generate_x_mark
from MLP_v3 import InitBlock, MHABlock, MLPBlock, Model, MultiFeatureBlock


class TestInitBlock(unittest.TestCase):
    def setUp(self):
        # Basic configurations
        self.batch_size = 8
        self.seq_len = 16
        self.input_dim = 10
        self.token_d_model = 32
        self.output_dim = 64
        self.token_conv_kernel = 5

    def test_init_block_with_pos_and_time(self):
        # Test case where both pos_d_model and time_d_model are provided
        pos_d_model = 16
        time_d_model = 16

        init_block = InitBlock(
            input_dim=self.input_dim,
            token_d_model=self.token_d_model,
            output_dim=self.output_dim,
            pos_d_model=pos_d_model,
            time_d_model=time_d_model,
            token_conv_kernel=self.token_conv_kernel,
            norm_type="batchnorm",
            activation_type="relu",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        x_mark = torch.randn(
            self.batch_size, self.seq_len, 4
        )  # Example for time features
        output = init_block(x, x_mark)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))

    def test_init_block_with_only_pos(self):
        # Test case where only pos_d_model is provided
        pos_d_model = 16

        init_block = InitBlock(
            input_dim=self.input_dim,
            token_d_model=self.token_d_model,
            output_dim=self.output_dim,
            pos_d_model=pos_d_model,
            token_conv_kernel=self.token_conv_kernel,
            norm_type="layernorm",
            activation_type="gelu",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = init_block(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))

    def test_init_block_with_only_time(self):
        # Test case where only time_d_model is provided
        time_d_model = 16

        init_block = InitBlock(
            input_dim=self.input_dim,
            token_d_model=self.token_d_model,
            output_dim=self.output_dim,
            time_d_model=time_d_model,
            token_conv_kernel=self.token_conv_kernel,
            norm_type="batchnorm",
            activation_type="gelu",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        x_mark = torch.randn(
            self.batch_size, self.seq_len, 4
        )  # Example for time features
        output = init_block(x, x_mark)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))

    def test_init_block_with_no_pos_and_time(self):
        # Test case where neither pos_d_model nor time_d_model are provided
        init_block = InitBlock(
            input_dim=self.input_dim,
            token_d_model=self.token_d_model,
            output_dim=self.output_dim,
            token_conv_kernel=self.token_conv_kernel,
            norm_type="batchnorm",
            activation_type="relu",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = init_block(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))


# class TestConv1DBlock(unittest.TestCase):
#     def setUp(self):
#         # Basic configurations
#         self.batch_size = 8
#         self.seq_len = 16
#         self.input_dim = 10
#         self.output_dim = 20

#         # Different kernel sizes to test
#         self.kernel_sizes = [5, 7, 9]

#         # Stride is fixed in this test
#         self.stride = 1

#         # Normalization types to test
#         self.norm_types = ["batchnorm", "layernorm"]

#     def test_conv1d_block_combinations(self):
#         for kernel_size in self.kernel_sizes:
#             for norm_type in self.norm_types:
#                 with self.subTest(kernel_size=kernel_size, norm_type=norm_type):
#                     # Initialize Conv1DBlock with the current combination
#                     conv_block = Conv1DBlock(
#                         input_dim=self.input_dim,
#                         output_dim=self.output_dim,
#                         kernel_size=kernel_size,
#                         stride=self.stride,
#                         norm_type=norm_type,
#                         activation_type="gelu",
#                     )

#                     # Create a random input tensor
#                     x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

#                     # Pass the input through the Conv1DBlock
#                     output = conv_block(x)

#                     # Check if the output shape is as expected
#                     self.assertEqual(
#                         output.shape, (self.batch_size, self.seq_len, self.output_dim)
#                     )

#     def test_conv1d_block_with_different_input_output_dims(self):
#         # Test cases with varying input and output dimensions
#         configurations = [
#             {"input_dim": 16, "output_dim": 32},
#             {"input_dim": 32, "output_dim": 64},
#             {"input_dim": 64, "output_dim": 128},
#         ]

#         for config in configurations:
#             for kernel_size in self.kernel_sizes:
#                 for norm_type in self.norm_types:
#                     with self.subTest(
#                         config=config, kernel_size=kernel_size, norm_type=norm_type
#                     ):
#                         # Initialize Conv1DBlock with the current configuration
#                         conv_block = Conv1DBlock(
#                             input_dim=config["input_dim"],
#                             output_dim=config["output_dim"],
#                             kernel_size=kernel_size,
#                             stride=self.stride,
#                             norm_type=norm_type,
#                             activation_type="gelu",
#                         )

#                         # Create a random input tensor
#                         x = torch.randn(
#                             self.batch_size, self.seq_len, config["input_dim"]
#                         )

#                         # Pass the input through the Conv1DBlock
#                         output = conv_block(x)

#                         # Check if the output shape is as expected
#                         self.assertEqual(
#                             output.shape,
#                             (self.batch_size, self.seq_len, config["output_dim"]),
#                         )

#     def test_conv1d_block_with_varied_seq_len(self):
#         # Test cases with different sequence lengths
#         seq_lens = [8, 16, 32]

#         for seq_len in seq_lens:
#             for kernel_size in self.kernel_sizes:
#                 for norm_type in self.norm_types:
#                     with self.subTest(
#                         seq_len=seq_len, kernel_size=kernel_size, norm_type=norm_type
#                     ):
#                         # Initialize Conv1DBlock with the current combination
#                         conv_block = Conv1DBlock(
#                             input_dim=self.input_dim,
#                             output_dim=self.output_dim,
#                             kernel_size=kernel_size,
#                             stride=self.stride,
#                             norm_type=norm_type,
#                             activation_type="gelu",
#                         )

#                         # Create a random input tensor with varying sequence lengths
#                         x = torch.randn(self.batch_size, seq_len, self.input_dim)

#                         # Pass the input through the Conv1DBlock
#                         output = conv_block(x)

#                         # Check if the output shape is as expected
#                         self.assertEqual(
#                             output.shape, (self.batch_size, seq_len, self.output_dim)
#                         )


class TestMLPBlock(unittest.TestCase):
    def test_mlpblock_hidden_dim_128_output_dim_64(self):
        batch_size, seq_len, input_dim = 32, 10, 256
        hidden_dim = 128
        output_dim = 64

        model = MLPBlock(input_dim, hidden_dim, output_dim, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_mlpblock_hidden_dim_256_output_dim_128(self):
        batch_size, seq_len, input_dim = 32, 20, 512
        hidden_dim = 256
        output_dim = 128

        model = MLPBlock(input_dim, hidden_dim, output_dim, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_mlpblock_hidden_dim_64_output_dim_32(self):
        batch_size, seq_len, input_dim = 16, 15, 128
        hidden_dim = 64
        output_dim = 32

        model = MLPBlock(input_dim, hidden_dim, output_dim, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_mlpblock_large_input(self):
        batch_size, seq_len, input_dim = 64, 50, 1024
        hidden_dim = 512
        output_dim = 256

        model = MLPBlock(input_dim, hidden_dim, output_dim, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))


class TestMHABlock(unittest.TestCase):
    def test_mhablock_num_heads_8(self):
        batch_size, seq_len, input_dim, output_dim = 32, 10, 128, 128
        num_heads = 8

        model = MHABlock(input_dim, output_dim, num_heads, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_mhablock_num_heads_16(self):
        batch_size, seq_len, input_dim, output_dim = 16, 20, 256, 256
        num_heads = 16

        model = MHABlock(input_dim, output_dim, num_heads, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_mhablock_num_heads_4(self):
        batch_size, seq_len, input_dim, output_dim = 8, 15, 64, 64
        num_heads = 4

        model = MHABlock(input_dim, output_dim, num_heads, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_mhablock_large_input(self):
        batch_size, seq_len, input_dim, output_dim = 64, 50, 512, 512
        num_heads = 8

        model = MHABlock(input_dim, output_dim, num_heads, activation_type="gelu")
        x = torch.randn(batch_size, seq_len, input_dim)

        output = model.forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))


class TestMultiFeatureBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 20
        self.num_heads = 4
        self.activation_type = "gelu"
        self.norm_type = "batchnorm"

    def test_multi_feature_block_varied_dimensions(self):
        configurations = [
            # Case 1: Standard configuration with matching dimensions
            {
                "input_dim": 64,
                "hidden_dim": 128,
                "output_dim": 64,
                "feat_conv_k_size": 3,
            },
            # Case 2: Larger hidden dimension
            {
                "input_dim": 64,
                "hidden_dim": 256,
                "output_dim": 128,
                "feat_conv_k_size": 5,
            },
            # Case 3: Different output dimension
            {
                "input_dim": 128,
                "hidden_dim": 256,
                "output_dim": 64,
                "feat_conv_k_size": 3,
            },
            # Case 4: Matching input and output, different hidden
            {
                "input_dim": 64,
                "hidden_dim": 128,
                "output_dim": 64,
                "feat_conv_k_size": 7,
            },
            # Case 5: Larger kernel size
            {
                "input_dim": 32,
                "hidden_dim": 64,
                "output_dim": 32,
                "feat_conv_k_size": 11,
            },
        ]

        for idx, config in enumerate(configurations):
            with self.subTest(i=idx):
                block = MultiFeatureBlock(
                    input_dim=config["input_dim"],
                    hidden_dim=config["hidden_dim"],
                    output_dim=config["output_dim"],
                    num_heads=self.num_heads,
                    feat_conv_k_size=config["feat_conv_k_size"],
                    stride=1,
                    norm_type=self.norm_type,
                    activation_type=self.activation_type,
                    norm_after_dict={"conv": True, "mha": True, "mlp": True},
                    skip_connection_mode="full",  # Test with full skip connections
                )

                # Create dummy input tensor
                x = torch.randn(self.batch_size, self.seq_len, config["input_dim"])

                # Forward pass
                output = block(x)

                # Check output dimensions
                self.assertEqual(
                    output.shape,
                    (self.batch_size, self.seq_len, config["output_dim"]),
                    f"Failed for config: {config}",
                )

    def test_multi_feature_block_no_norm_after_mha(self):
        # Test case where normalization is not applied after MHA
        block = MultiFeatureBlock(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=self.num_heads,
            feat_conv_k_size=3,
            stride=1,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            norm_after_dict={"conv": True, "mha": False, "mlp": True},
            skip_connection_mode="full",  # Test with full skip connections
        )

        # Create dummy input tensor
        x = torch.randn(self.batch_size, self.seq_len, 64)

        # Forward pass
        output = block(x)

        # Check output dimensions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))

    def test_multi_feature_block_no_norm_after_mlp(self):
        # Test case where normalization is not applied after MLP
        block = MultiFeatureBlock(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=self.num_heads,
            feat_conv_k_size=3,
            stride=1,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            norm_after_dict={"conv": True, "mha": True, "mlp": False},
            skip_connection_mode="full",  # Test with full skip connections
        )

        # Create dummy input tensor
        x = torch.randn(self.batch_size, self.seq_len, 64)

        # Forward pass
        output = block(x)

        # Check output dimensions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))

    def test_multi_feature_block_no_skip_connections(self):
        # Test case with no skip connections
        block = MultiFeatureBlock(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=self.num_heads,
            feat_conv_k_size=3,
            stride=1,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            norm_after_dict={"conv": True, "mha": True, "mlp": True},
            skip_connection_mode="none",  # No skip connections
        )

        # Create dummy input tensor
        x = torch.randn(self.batch_size, self.seq_len, 64)

        # Forward pass
        output = block(x)

        # Check output dimensions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))

    def test_multi_feature_block_skip_connections_conv_mha(self):
        # Test case with skip connections after Conv1D and MHA blocks
        block = MultiFeatureBlock(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=self.num_heads,
            feat_conv_k_size=3,
            stride=1,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            norm_after_dict={"conv": True, "mha": True, "mlp": True},
            skip_connection_mode="conv_mha",  # Skip connections after Conv1D and MHA
        )

        # Create dummy input tensor
        x = torch.randn(self.batch_size, self.seq_len, 64)

        # Forward pass
        output = block(x)

        # Check output dimensions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))

    def test_multi_feature_block_skip_connections_conv_mlp(self):
        # Test case with skip connections after Conv1D and MLP blocks
        block = MultiFeatureBlock(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=self.num_heads,
            feat_conv_k_size=3,
            stride=1,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            norm_after_dict={"conv": True, "mha": True, "mlp": True},
            skip_connection_mode="conv_mlp",  # Skip connections after Conv1D and MLP
        )

        # Create dummy input tensor
        x = torch.randn(self.batch_size, self.seq_len, 64)

        # Forward pass
        output = block(x)

        # Check output dimensions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))

    def test_multi_feature_block_skip_connections_full(self):
        # Test case with full skip connections
        block = MultiFeatureBlock(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=self.num_heads,
            feat_conv_k_size=3,
            stride=1,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            norm_after_dict={"conv": True, "mha": True, "mlp": True},
            skip_connection_mode="full",  # Full skip connections
        )

        # Create dummy input tensor
        x = torch.randn(self.batch_size, self.seq_len, 64)

        # Forward pass
        output = block(x)

        # Check output dimensions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))


class TestModel(unittest.TestCase):
    def setUp(self):
        # Set up basic configuration for the model
        self.config = argparse.Namespace(
            input_dim=84,
            token_d_model=32,
            pos_d_model=64,
            time_d_model=64,
            conv_out_dim=128,
            d_model=256,
            hidden_d_model=128,
            last_d_model=64,
            e_layers=4,
            num_heads=8,
            norm_type="layernorm",
            activation_type="gelu",
            pred_len=1,
            output_dim=1,
            token_conv_kernel=5,
        )

        # Create a batch of example inputs
        self.batch_size = 16
        self.seq_len = 36
        self.num_features = 84

        self.x = torch.randn(self.batch_size, self.seq_len, self.num_features)

        # Generate x_mark using the provided function
        self.x_mark = generate_x_mark(self.batch_size, self.seq_len)

    def test_forward_pass(self):
        # Initialize the model with the configuration
        model = Model(self.config)

        # Run the forward pass
        output = model(self.x, self.x_mark)

        # Assert output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.config.pred_len, self.config.output_dim),
        )
        print(f"Output shape: {output.shape}")

    def test_various_configurations(self):
        # Test with different configurations
        configurations = [
            # Varying d_model, hidden_d_model, and last_d_model
            {"d_model": 128, "hidden_d_model": 64, "last_d_model": 32},
            {"d_model": 256, "hidden_d_model": 128, "last_d_model": 64},
            {"d_model": 512, "hidden_d_model": 256, "last_d_model": 128},
            # Varying e_layers, num_heads, and conv_out_dim
            {"e_layers": 2, "num_heads": 2, "conv_out_dim": 64},
            {"e_layers": 4, "num_heads": 4, "conv_out_dim": 128},
            {"e_layers": 6, "num_heads": 8, "conv_out_dim": 256},
            # Varying multiple parameters simultaneously
            {"d_model": 128, "e_layers": 2, "num_heads": 2, "conv_out_dim": 64},
            {"d_model": 256, "e_layers": 4, "num_heads": 4, "conv_out_dim": 128},
            {"d_model": 512, "e_layers": 6, "num_heads": 8, "conv_out_dim": 256},
            {
                "d_model": 128,
                "hidden_d_model": 128,
                "last_d_model": 128,
                "e_layers": 3,
                "num_heads": 4,
                "conv_out_dim": 128,
            },
            {
                "d_model": 256,
                "hidden_d_model": 256,
                "last_d_model": 256,
                "e_layers": 5,
                "num_heads": 8,
                "conv_out_dim": 256,
            },
            {
                "d_model": 512,
                "hidden_d_model": 512,
                "last_d_model": 512,
                "e_layers": 7,
                "num_heads": 16,
                "conv_out_dim": 512,
            },
            # Varying only one parameter while keeping others constant
            {
                "d_model": 256,
                "hidden_d_model": 128,
                "last_d_model": 64,
                "e_layers": 4,
                "num_heads": 4,
                "conv_out_dim": 128,
            },
            {
                "d_model": 256,
                "hidden_d_model": 128,
                "last_d_model": 64,
                "e_layers": 4,
                "num_heads": 8,
                "conv_out_dim": 128,
            },
            {
                "d_model": 256,
                "hidden_d_model": 128,
                "last_d_model": 64,
                "e_layers": 4,
                "num_heads": 16,
                "conv_out_dim": 128,
            },
            {
                "d_model": 256,
                "hidden_d_model": 128,
                "last_d_model": 64,
                "e_layers": 6,
                "num_heads": 4,
                "conv_out_dim": 128,
            },
            {
                "d_model": 256,
                "hidden_d_model": 128,
                "last_d_model": 64,
                "e_layers": 8,
                "num_heads": 4,
                "conv_out_dim": 128,
            },
        ]

        for cfg in configurations:
            for key, value in cfg.items():
                setattr(self.config, key, value)

            model = Model(self.config)
            output = model(self.x, self.x_mark)

            # Assert output shape
            self.assertEqual(
                output.shape,
                (self.batch_size, self.config.pred_len, self.config.output_dim),
            )
            print(
                f"Test passed for configuration: {cfg} with output shape: {output.shape}"
            )


if __name__ == "__main__":
    unittest.main()
