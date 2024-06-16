import torch

# Sample tensors
preds = torch.randn(100, 10, requires_grad=True)  # Example prediction tensor
tgt = torch.randn(100, 10, requires_grad=True)    # Example target tensor
logs = [0, 2, 4]                                 # Indices to be transformed
log_shifts = torch.randn(10)                     # Example log shifts

# Convert the list of indices to a boolean mask
mask = torch.zeros(preds.shape[1], dtype=torch.bool)
mask[logs] = True

# Apply the exponential transformation and shift
exp_preds = torch.exp(preds[:, mask]) - log_shifts[mask]
exp_tgt = torch.exp(tgt[:, mask]) - log_shifts[mask]

# Create copies of preds and tgt to avoid in-place operations
new_preds = preds.clone()
new_tgt = tgt.clone()

# Update the selected columns in new_preds and new_tgt
new_preds[:, mask] = exp_preds
new_tgt[:, mask] = exp_tgt

# Example backward pass
loss = torch.mean(new_preds + new_tgt)  # Example loss function
loss.backward()  # Perform backpropagation

print(new_preds)
print(new_tgt)
