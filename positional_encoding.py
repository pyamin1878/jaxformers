import jax.numpy as jnp

def positional_encoding(seq_length: int, d_model: int) -> jnp.ndarray:
    positions = jnp.arange(seq_length)[:, jnp.newaxis]
    dimensions = jnp.arange(d_model)[jnp.newaxis, :]
    angles = positions / jnp.power(10000, (2 * (dimensions // 2)) / d_model)
    pos_encoding = jnp.zeros((seq_length, d_model))
    pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(angles[:, 0::2]))
    pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(angles[:, 1::2]))
    return pos_encoding