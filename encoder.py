import jax.numpy as jnp
from attention import scaled_dot_product_attention
from positional_encoding import positional_encoding
from typing import Union, Any, Optional

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k - d_model // num_heads
    
    def split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_length, _ = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
    
#   def __call__(self, *args: Any, **kwds: Any) -> Any:

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch_size, seq_length, _ = query.shape

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
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
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = jnp.zeros((d_model,))
        self.layer_norm2 = jnp.zeros((d_model,))
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:


