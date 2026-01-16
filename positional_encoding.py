import jax.numpy as jnp


def positional_encoding(seq_length: int, d_model: int) -> jnp.ndarray:
    """
    Generate sinusoidal positional encodings.

    Uses the formulas from "Attention Is All You Need":
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the model

    Returns:
        Positional encoding tensor of shape (seq_length, d_model)
    """
    positions = jnp.arange(seq_length)[:, jnp.newaxis]
    dimensions = jnp.arange(d_model)[jnp.newaxis, :]

    # Compute the angle rates
    angle_rates = 1 / jnp.power(10000.0, (2 * (dimensions // 2)) / d_model)

    # Compute the angles
    angles = positions * angle_rates

    # Apply sin to even indices, cos to odd indices
    encodings = jnp.where(
        dimensions % 2 == 0,
        jnp.sin(angles),
        jnp.cos(angles)
    )

    return encodings
