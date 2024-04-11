import jax.numpy as jnp
from typing import Optional

def scaled_dot_product_attention(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)
    
    if mask is not None:
        scores = jnp.where(mask == 0, -jnp.inf, scores)
    attention_weights = jnp.softmax(scores, axis=-1)
    output = jnp.matmul(attention_weights, value)
    
    return output, attention_weights