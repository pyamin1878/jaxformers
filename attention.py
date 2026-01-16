import jax.numpy as jnp
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_v, d_v) where seq_len_k == seq_len_v
        mask: Optional mask tensor broadcastable to (..., seq_len_q, seq_len_k)

    Returns:
        Tuple of (attention_output, attention_weights)
        - attention_output: shape (..., seq_len_q, d_v)
        - attention_weights: shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)

    # Compute attention weights via softmax
    attention_weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / jnp.sum(attention_weights, axis=-1, keepdims=True)

    # Compute attention output
    attention_output = jnp.matmul(attention_weights, value)

    return attention_output, attention_weights
