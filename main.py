import jax.numpy as jnp
from jax import random
from transformer_layer import transformer_layer

# Initialize random key
key = random.PRNGKey(0)

# Define input data and hyperparameters
batch_size = 2
seq_length = 5
d_model = 64
num_heads = 8
d_ff = 128

# Generate random input data
input_data = random.normal(key, (batch_size, seq_length, d_model))

# Pass the input through the transformer layer
output = transformer_layer(input_data, num_heads, d_model, d_ff)

print("Input shape:", input_data.shape)
print("Output shape:", output.shape)