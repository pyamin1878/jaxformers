import jax.numpy as jnp
from attention import scaled_dot_product_attention
from positional_encoding import positional_encoding
from typing import Optional

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
    
class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = jnp.zeros((d_model,))
        self.layer_norm2 = jnp.zeros((d_model,))
        self.layer_norm3 = jnp.zeros((d_model,))
        self.dropout_rate = dropout_rate
        
    def __call__(
        self, x: jnp.ndarray, encoder_output: jnp.ndarray, src_mask: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        attention1 = self.masked_multi_head_attention(x, x, x, tgt_mask)
        x = x + attention1
        x = jnp.dot(x, self.layer_norm1)
        
        attention2 = self.multi_head_attention(x, encoder_output, encoder_output, src_mask)
        x = x + attention2
        x = jnp.dot(x, self.layer_norm2)
        
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = jnp.dot(x, self.layer_norm3)
        
        x = jnp.where(jnp.random.uniform(x.shape) < self.dropout_rate, 0, x)
        
        return x

class Decoder:
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        
    def __call__(
        self, x: jnp.ndarray, encoder_output: jnp.ndarray, src_mask: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        seq_length, _ = x.shape
        x += positional_encoding(seq_length, x.shape[-1])
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x