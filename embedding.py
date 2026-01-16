import jax.numpy as jnp
from jax import random
from positional_encoding import positional_encoding
from typing import Optional


class TokenEmbedding:
    """Converts token indices to dense vectors."""

    def __init__(self, vocab_size: int, d_model: int, key: random.PRNGKey):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            key: JAX random key for initialization
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Xavier/Glorot initialization
        scale = jnp.sqrt(2.0 / (vocab_size + d_model))
        self.embedding = random.normal(key, (vocab_size, d_model)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Token indices of shape (batch_size, seq_length)

        Returns:
            Embeddings of shape (batch_size, seq_length, d_model)
        """
        return self.embedding[x] * jnp.sqrt(self.d_model)


class LearnablePositionalEncoding:
    """Learnable positional encodings."""

    def __init__(self, max_seq_length: int, d_model: int, key: random.PRNGKey):
        """
        Args:
            max_seq_length: Maximum sequence length
            d_model: Dimension of the model
            key: JAX random key for initialization
        """
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        # Initialize with small random values
        self.encoding = random.normal(key, (max_seq_length, d_model)) * 0.02

    def __call__(self, seq_length: int) -> jnp.ndarray:
        """
        Args:
            seq_length: Length of the sequence

        Returns:
            Positional encodings of shape (seq_length, d_model)
        """
        return self.encoding[:seq_length]


class TransformerEmbedding:
    """Combined token embedding and positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_length: int,
        key: random.PRNGKey,
        learnable_pos: bool = False,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length
            key: JAX random key for initialization
            learnable_pos: Whether to use learnable positional encodings
            dropout_rate: Dropout rate
        """
        key1, key2 = random.split(key)
        self.token_embedding = TokenEmbedding(vocab_size, d_model, key1)
        self.learnable_pos = learnable_pos
        self.dropout_rate = dropout_rate

        if learnable_pos:
            self.pos_encoding = LearnablePositionalEncoding(max_seq_length, d_model, key2)
        else:
            self.pos_encoding = None

    def __call__(
        self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Args:
            x: Token indices of shape (batch_size, seq_length)
            key: Optional JAX random key for dropout

        Returns:
            Embeddings of shape (batch_size, seq_length, d_model)
        """
        seq_length = x.shape[1]
        d_model = self.token_embedding.d_model

        # Token embeddings
        embeddings = self.token_embedding(x)

        # Add positional encodings
        if self.learnable_pos:
            embeddings = embeddings + self.pos_encoding(seq_length)
        else:
            embeddings = embeddings + positional_encoding(seq_length, d_model)

        # Apply dropout if key is provided
        if key is not None and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, embeddings.shape)
            embeddings = jnp.where(mask, embeddings / keep_prob, 0)

        return embeddings
