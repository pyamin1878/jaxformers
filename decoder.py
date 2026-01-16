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


def create_causal_mask(seq_length: int) -> jnp.ndarray:
    """Create a causal mask to prevent attending to future positions."""
    mask = jnp.tril(jnp.ones((seq_length, seq_length)))
    return mask


class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_length, _ = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        batch_size, seq_length, _ = query.shape

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.d_model
        )

        return scaled_attention


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, key: random.PRNGKey):
        key1, key2 = random.split(key)
        # Xavier initialization
        self.layer1 = random.normal(key1, (d_model, d_ff)) * jnp.sqrt(2.0 / (d_model + d_ff))
        self.layer2 = random.normal(key2, (d_ff, d_model)) * jnp.sqrt(2.0 / (d_ff + d_model))
        self.bias1 = jnp.zeros((d_ff,))
        self.bias2 = jnp.zeros((d_model,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.dot(x, self.layer1) + self.bias1
        x = jnp.maximum(x, 0)  # ReLU
        x = jnp.dot(x, self.layer2) + self.bias2
        return x


class DecoderLayer:
    """Single decoder layer with masked self-attention, cross-attention, and FFN."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, key: random.PRNGKey, dropout_rate: float = 0.1
    ):
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, key)
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        self_attn_mask: Optional[jnp.ndarray] = None,
        cross_attn_mask: Optional[jnp.ndarray] = None,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attn_mask: Mask for self-attention (causal mask)
            cross_attn_mask: Mask for cross-attention (padding mask)
            key: Optional random key for dropout

        Returns:
            Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual and layer norm
        self_attn_output = self.masked_self_attention(x, x, x, self_attn_mask)
        x = layer_norm(x + self_attn_output)

        # Cross-attention with residual and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = layer_norm(x + cross_attn_output)

        # Feed-forward with residual and layer norm
        ff_output = self.feed_forward(x)
        x = layer_norm(x + ff_output)

        # Apply dropout if key is provided
        if key is not None and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0)

        return x


class Decoder:
    """Transformer decoder with multiple decoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        key: random.PRNGKey,
        dropout_rate: float = 0.1,
    ):
        keys = random.split(key, num_layers)
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, keys[i], dropout_rate)
            for i in range(num_layers)
        ]
        self.d_model = d_model

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        self_attn_mask: Optional[jnp.ndarray] = None,
        cross_attn_mask: Optional[jnp.ndarray] = None,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attn_mask: Mask for self-attention (causal mask)
            cross_attn_mask: Mask for cross-attention (padding mask)
            key: Optional random key for dropout

        Returns:
            Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Add positional encoding
        seq_length = x.shape[1]
        x = x + positional_encoding(seq_length, self.d_model)

        # Create causal mask if not provided
        if self_attn_mask is None:
            self_attn_mask = create_causal_mask(seq_length)

        for i, layer in enumerate(self.layers):
            layer_key = random.fold_in(key, i) if key is not None else None
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask, layer_key)

        return x
