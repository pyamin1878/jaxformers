import jax.numpy as jnp
from transformer import Transformer

# Define the model hyperparameters
num_encoder_layers = 6
num_decoder_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
input_vocab_size = 1000
target_vocab_size = 1000
dropout_rate = 0.1

# Create an instance of the Transformer model
transformer = Transformer(
    num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff,
    input_vocab_size, target_vocab_size, dropout_rate
)

# Generate random input data for demonstration
batch_size = 2
src_seq_length = 10
tgt_seq_length = 8
src = jnp.random.randint(0, input_vocab_size, (batch_size, src_seq_length))
tgt = jnp.random.randint(0, target_vocab_size, (batch_size, tgt_seq_length))

# Forward pass through the Transformer model
output = transformer(src, tgt)

print("Input shape:", src.shape)
print("Output shape:", output.shape)