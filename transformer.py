import jax.numpy as jnp
from encoder import Encoder
from decoder import Decoder
from embedding import Embedding
from typing import Optional

class Transformer:
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, d_model: int, num_heads: int, d_ff: int,
        input_vocab_size: int, target_vocab_size: int, dropout_rate: float
    ):
        self.encoder_embedding = Embedding(input_vocab_size, d_model)
        self.decoder_embedding = Embedding(input_vocab_size, d_model)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout_rate)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout_rate)
        self.final_linear = jnp.zeros((d_model, target_vocab_size))

    def __call__(self, src: jnp.ndarray, tgt: jnp.ndarray, src_mask: Optional[jnp.ndarray] = None,
                 tgt_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        src_embedded = self.encoder_embedding(src)
        tgt_embedded = self.decoder_embedding(tgt)

        encoder_output = self.encoder(src_embedded, src_mask)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        output = jnp.dot(decoder_output, self.final_linear)
        output = jnp.softmax(output, axis=-1)

        return output