import numpy as np
import pandas as pd
import torch

from seq2seq_utils import to_tensor


# Test the function
def test_to_tensor():
    # Create synthetic data for testing
    batch_size = 3
    sequence_length = 5
    seq_variables_x = ['seq_var1', 'seq_var2']
    scalar_variables_x = ['scalar_var1', 'scalar_var2']

    # Generate synthetic data
    data = {
        'seq_var1_0': [1, 2, 3],
        'seq_var1_1': [4, 5, 6],
        'seq_var1_2': [7, 8, 9],
        'seq_var1_3': [10, 11, 12],
        'seq_var1_4': [13, 14, 15],

        'seq_var2_0': [16, 17, 18],
        'seq_var2_1': [19, 20, 21],
        'seq_var2_2': [22, 23, 24],
        'seq_var2_3': [25, 26, 27],
        'seq_var2_4': [28, 29, 30],

        'scalar_var1': [31, 32, 33],
        'scalar_var2': [34, 35, 36]
    }

    df = pd.DataFrame(data)

    # Expected shape of the output tensor
    expected_shape = (batch_size, sequence_length, len(seq_variables_x) + len(scalar_variables_x))

    # Call the function
    tensor_data = to_tensor(df, batch_size, sequence_length, seq_variables_x, scalar_variables_x)

    # Assertions
    assert tensor_data.shape == expected_shape, f"Expected shape {expected_shape}, but got {tensor_data.shape}"
    assert tensor_data.dtype == torch.float32, f"Expected dtype torch.float32, but got {tensor_data.dtype}"

    for i in range(tensor_data.shape[0]):
        print(tensor_data[i])

    print("Test passed!")


# Run the test
test_to_tensor()

