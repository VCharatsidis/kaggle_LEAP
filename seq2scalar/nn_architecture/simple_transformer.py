import torch
import torch.nn as nn
import math

class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()

        self.output_dim = output_dim
        self.src_input_linear = nn.Linear(input_dim, d_model)
        self.tgt_input_linear = nn.Linear(output_dim, d_model)
        self.output_linear = nn.Linear(d_model, output_dim)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Transform input and target to d_model dimensions
        src = self.src_input_linear(src)
        tgt = self.tgt_input_linear(tgt)

        # Apply the transformer
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)

        # Transform the output to the desired dimension
        output = self.output_linear(output)
        return output

    def generate(self, src, start_symbol, max_length):
        # Prepare the memory from the source
        src = self.src_input_linear(src)
        memory = self.transformer.encoder(src)

        # Start with a tensor for the start_symbol
        generated = torch.tensor([[start_symbol]], dtype=torch.long, device=src.device)
        generated = self.tgt_input_linear(F.one_hot(generated, num_classes=self.output_dim).float())

        for _ in range(max_length - 1):  # -1 because the first token is already included
            tgt_mask = self.transformer.generate_square_subsequent_mask(generated.size(0)).to(generated.device)
            out = self.transformer.decoder(generated, memory, tgt_mask=tgt_mask)
            out = self.output_linear(out)
            next_token = out[-1, :, :].argmax(dim=-1, keepdim=True)
            next_token = self.tgt_input_linear(F.one_hot(next_token, num_classes=self.output_dim).float())

            # Append the predicted token to the generated sequence
            generated = torch.cat([generated, next_token], dim=0)

        return generated

    # def generate(self, src, max_len):
    #     src = self.src_input_linear(src)
    #     memory = self.transformer.encoder(src)
    #
    #     # Prepare initial input (e.g., zeros or start token)
    #     tgt = torch.zeros(src.size(0), 1, self.d_model).to(src.device)  # Start with a single time step
    #     outputs = []
    #
    #     for _ in range(max_len):
    #         tgt_pos_enc = self.pos_decoder(tgt)
    #         output = self.decoder(tgt_pos_enc, memory)
    #         output = self.output_fc(output[:, -1, :])  # Get the last time step and reshape to [batch_size, output_dim]
    #
    #         outputs.append(output.unsqueeze(1))  # Add a new dimension to match [batch_size, 1, output_dim]
    #         new_tgt = self.input_fc_tgt(output).unsqueeze(1)  # Transform output back to tgt space and add dimension
    #         tgt = torch.cat([tgt, new_tgt], dim=1)  # Append the generated output to the target sequence
    #
    #     outputs = torch.cat(outputs, dim=1)  # Concatenate along the sequence dimension
    #     return outputs
