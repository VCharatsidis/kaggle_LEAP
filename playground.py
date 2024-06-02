import torch

# Example tensor with shape (16, 60, 14)
tensor_data = torch.rand(16, 60, 14)

# Number of variables for which we need to compute the mean along dim=2
scalar_vars_num = 6
vector_vars_num = 8

# Slice the tensor to get the first 6 variables and the last 8 variables
vector_vars = tensor_data[:, :, :scalar_vars_num]
scalar_vars = tensor_data[:, :, -vector_vars_num:]

# Compute the mean along dim=1, excluding the first 6 that are vector vars
mean_vars = scalar_vars.mean(dim=1, keepdim=True).squeeze(dim=1)

print(vector_vars.shape, mean_vars.shape)  # Should print (16, 60, 6) (16, 1, 8
vector_vars = vector_vars.reshape(vector_vars.size(0), -1)  # shape: [batch_size, seq_len * d_model]

print(vector_vars.shape, mean_vars.shape)  # Should print (16, 60, 6
# Concatenate the first variables and the mean variables along dim=2
concatenated = torch.cat((vector_vars, mean_vars), dim=1)

# Flatten the tensor along the second and third dimensions
flattened = concatenated.view(concatenated.size(0), -1)

print(flattened.shape)  # Should print (16, 60*6 +
