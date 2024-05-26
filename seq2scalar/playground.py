# Example parameters
import torch

from seq_to_scalars_transformer_custom import SequenceToScalarTransformer_custom
from seq_to_scalars_transformer import SequenceToScalarTransformer

input_dim = 556  # Number of input variables
output_dim = 368  # Number of output variables
d_model = 512  # Dimension of the model's internal representation
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
dropout = 0.1

# Initialize the model
model = SequenceToScalarTransformer(input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
model_2 = SequenceToScalarTransformer_custom(input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
model.eval()
model_2.eval()
# Example input data
batch_size = 50
seq_len = 60  # Number of vertical levels

# Create random input tensor for demonstration
src = torch.randn(seq_len, batch_size, input_dim)  # [seq_len, batch_size, input_dim]

print(src.shape)
# Forward pass
output = model(src)
print(output.shape)  # Should output: torch.Size([batch_size, output_dim])

output_2 = model_2(src)

close = torch.isclose(output, output_2, rtol=1e-03, atol=1e-03)
print(close)
all_close = torch.all(close)
print("Are all elements close?", all_close)