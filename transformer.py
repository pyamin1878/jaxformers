import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple

from embedding import TransformerEmbedding
from encoder import Encoder
from decoder import Decoder, create_causal_mask


class OutputLayer:
    """Final linear layer with softmax for output prediction."""

    def __init__(self, d_model: int, vocab_size: int, key: random.PRNGKey):
        """
        Args:
            d_model: Dimension of the model
            vocab_size: Size of the output vocabulary
            key: JAX random key for initialization
        """
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (d_model + vocab_size))
        self.weight = random.normal(key, (d_model, vocab_size)) * scale
        self.bias = jnp.zeros((vocab_size,))

    def __call__(self, x: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        """
        Args:
            x: Input of shape (batch_size, seq_length, d_model)
            temperature: Softmax temperature for controlling randomness

        Returns:
            Log probabilities of shape (batch_size, seq_length, vocab_size)
        """
        logits = jnp.dot(x, self.weight) + self.bias
        logits = logits / temperature
        # Log-softmax for numerical stability
        log_probs = logits - jnp.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - jnp.log(jnp.sum(jnp.exp(log_probs), axis=-1, keepdims=True))
        return log_probs

    def logits(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return raw logits without softmax."""
        return jnp.dot(x, self.weight) + self.bias


class Transformer:
    """Full Transformer model for sequence-to-sequence tasks."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout_rate: float = 0.1,
        learnable_pos: bool = False,
        key: random.PRNGKey = None,
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate
            learnable_pos: Whether to use learnable positional encodings
            key: JAX random key for initialization
        """
        if key is None:
            key = random.PRNGKey(0)

        keys = random.split(key, 5)

        self.d_model = d_model
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_seq_length, keys[0], learnable_pos, dropout_rate
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_seq_length, keys[1], learnable_pos, dropout_rate
        )
        self.encoder = Encoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout_rate
        )
        self.decoder = Decoder(
            num_decoder_layers, d_model, num_heads, d_ff, keys[3], dropout_rate
        )
        self.output_layer = OutputLayer(d_model, tgt_vocab_size, keys[4])

    def encode(
        self,
        src: jnp.ndarray,
        src_mask: Optional[jnp.ndarray] = None,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Encode source sequence.

        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            src_mask: Optional source mask
            key: Optional random key for dropout

        Returns:
            Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        keys = random.split(key, 2) if key is not None else [None, None]
        src_emb = self.src_embedding(src, keys[0])
        return self.encoder(src_emb, src_mask, keys[1])

    def decode(
        self,
        tgt: jnp.ndarray,
        encoder_output: jnp.ndarray,
        tgt_mask: Optional[jnp.ndarray] = None,
        src_tgt_mask: Optional[jnp.ndarray] = None,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Decode target sequence given encoder output.

        Args:
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Optional target mask (causal mask)
            src_tgt_mask: Optional source-target mask
            key: Optional random key for dropout

        Returns:
            Decoder output of shape (batch_size, tgt_seq_len, d_model)
        """
        keys = random.split(key, 2) if key is not None else [None, None]
        tgt_emb = self.tgt_embedding(tgt, keys[0])
        return self.decoder(tgt_emb, encoder_output, tgt_mask, src_tgt_mask, keys[1])

    def __call__(
        self,
        src: jnp.ndarray,
        tgt: jnp.ndarray,
        src_mask: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None,
        src_tgt_mask: Optional[jnp.ndarray] = None,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Full forward pass.

        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask (causal mask)
            src_tgt_mask: Optional source-target mask
            key: Optional random key for dropout

        Returns:
            Log probabilities of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        keys = random.split(key, 2) if key is not None else [None, None]

        encoder_output = self.encode(src, src_mask, keys[0])
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_tgt_mask, keys[1])
        return self.output_layer(decoder_output)

    def generate(
        self,
        src: jnp.ndarray,
        max_length: int,
        start_token: int,
        end_token: int,
        temperature: float = 1.0,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Generate output sequence autoregressively.

        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            max_length: Maximum length of generated sequence
            start_token: Start of sequence token ID
            end_token: End of sequence token ID
            temperature: Sampling temperature
            key: Random key for sampling

        Returns:
            Generated token indices of shape (batch_size, generated_length)
        """
        batch_size = src.shape[0]
        encoder_output = self.encode(src)

        # Start with start token
        generated = jnp.full((batch_size, 1), start_token, dtype=jnp.int32)

        for i in range(max_length - 1):
            # Get embeddings for generated sequence so far
            tgt_emb = self.tgt_embedding(generated)

            # Decode
            decoder_output = self.decoder(tgt_emb, encoder_output)

            # Get logits for last position
            logits = self.output_layer.logits(decoder_output[:, -1:, :])
            logits = logits / temperature

            # Sample or argmax
            if key is not None:
                key, subkey = random.split(key)
                probs = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
                probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
                next_token = random.categorical(subkey, jnp.log(probs), axis=-1)
            else:
                next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token.reshape(batch_size, 1)
            generated = jnp.concatenate([generated, next_token], axis=1)

        return generated


class TransformerLM:
    """Transformer decoder-only language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout_rate: float = 0.1,
        learnable_pos: bool = False,
        key: random.PRNGKey = None,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate
            learnable_pos: Whether to use learnable positional encodings
            key: JAX random key for initialization
        """
        if key is None:
            key = random.PRNGKey(0)

        keys = random.split(key, 3)

        self.d_model = d_model
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_seq_length, keys[0], learnable_pos, dropout_rate
        )
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, keys[1], dropout_rate)
        self.output_layer = OutputLayer(d_model, vocab_size, keys[2])

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Token indices of shape (batch_size, seq_length)
            mask: Optional causal mask
            key: Optional random key for dropout

        Returns:
            Log probabilities of shape (batch_size, seq_length, vocab_size)
        """
        keys = random.split(key, 2) if key is not None else [None, None]

        # Embed tokens
        embeddings = self.embedding(x, keys[0])

        # Create causal mask if not provided
        seq_length = x.shape[1]
        if mask is None:
            mask = create_causal_mask(seq_length)

        # Create dummy encoder output (zeros) for decoder-only mode
        # The decoder will only use self-attention due to how we set it up
        dummy_encoder_output = jnp.zeros_like(embeddings)

        # Pass through decoder (self-attention only effectively)
        decoder_output = self.decoder(embeddings, dummy_encoder_output, mask, None, keys[1])

        return self.output_layer(decoder_output)
