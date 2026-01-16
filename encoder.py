import jax.numpy as jnp
from jax import random
from attention import scaled_dot_product_attention
from positional_encoding import positional_encoding
from typing import Optional


def layer_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Apply layer normalization along the last dimension."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
    def split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_length, _ = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def __call__(
        self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        batch_size, seq_length, _ = query.shape
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)
        
        return scaled_attention

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.layer1 = jnp.zeros((d_model, d_ff))
        self.layer2 = jnp.zeros((d_ff, d_model))
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.dot(x, self.layer1)
        x = jnp.maximum(x, 0)
        x = jnp.dot(x, self.layer2)
        return x

class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout_rate = dropout_rate

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[random.PRNGKey] = None
    ) -> jnp.ndarray:
        # Multi-head self-attention with residual connection and layer norm
        attention_output = self.multi_head_attention(x, x, x, mask)
        x = layer_norm(x + attention_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = layer_norm(x + ff_output)

        # Apply dropout during training if key is provided
        if key is not None and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0)

        return x

class Encoder:
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[random.PRNGKey] = None
    ) -> jnp.ndarray:
        # x shape: (batch_size, seq_length, d_model)
        seq_length = x.shape[1]
        x = x + positional_encoding(seq_length, x.shape[-1])

        for i, layer in enumerate(self.layers):
            layer_key = random.fold_in(key, i) if key is not None else None
            x = layer(x, mask, layer_key)

        return x


