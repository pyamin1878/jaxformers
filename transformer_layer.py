import jax.numpy as jnp
from jax import random
from attention import scaled_dot_product_attention

def multi_head_attention(queries, keys, values, num_heads, d_model, mask=None):
    batch_size = queries.shape[0]
    seq_length = queries.shape[1]
    
    # Linear layers
    query = jnp.transpose(queries.reshape(batch_size, seq_length, num_heads, -1), (0, 2, 1, 3))
    key = jnp.transpose(keys.reshape(batch_size, seq_length, num_heads, -1), (0, 2, 1, 3))
    value = jnp.transpose(values.reshape(batch_size, seq_length, num_heads, -1), (0, 2, 1, 3))
    
    # Scaled dot-product attention
    scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)
    
    # Reshape and concatenate the heads
    scaled_attention = jnp.transpose(scaled_attention, (0, 2, 1, 3))
    concat_attention = scaled_attention.reshape(batch_size, seq_length, -1)
    
    return concat_attention, attention_weights

def transformer_layer(inputs, num_heads, d_model, d_ff, dropout_rate=0.1):
    # Multi-head attention
    attention_output, _ = multi_head_attention(inputs, inputs, inputs, num_heads, d_model)
    attention_output = inputs + attention_output
    
    # Feedforward
    ff_output = jnp.dot(attention_output, jnp.ones((d_model, d_ff)))
    ff_output = ff_output + jnp.ones((ff_output.shape[0], ff_output.shape[1], d_ff))
    ff_output = jnp.dot(ff_output, jnp.ones((d_ff, d_model)))
    ff_output = ff_output + attention_output
    
    return ff_output