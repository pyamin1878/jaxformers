"""
JAXformers Demo: Test all Transformer components.
"""

import jax.numpy as jnp
from jax import random

from attention import scaled_dot_product_attention
from positional_encoding import positional_encoding
from encoder import MultiHeadAttention, FeedForward, EncoderLayer, Encoder
from embedding import TokenEmbedding, LearnablePositionalEncoding, TransformerEmbedding
from decoder import Decoder, DecoderLayer, create_causal_mask
from transformer import Transformer, TransformerLM, OutputLayer


def main():
    # Configuration
    batch_size = 2
    seq_length = 10
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2
    vocab_size = 100

    # Initialize random key
    key = random.PRNGKey(42)

    # Generate random input
    key, subkey = random.split(key)
    x = random.normal(subkey, (batch_size, seq_length, d_model))

    print("=" * 60)
    print("JAXformers: Transformer Implementation Demo")
    print("=" * 60)

    # Test scaled dot-product attention
    print("\n1. Scaled Dot-Product Attention")
    print("-" * 50)
    query = key_tensor = value = x
    attention_output, attention_weights = scaled_dot_product_attention(query, key_tensor, value)
    print(f"   Input shape:             {x.shape}")
    print(f"   Attention output shape:  {attention_output.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")

    # Test positional encoding
    print("\n2. Positional Encoding (Sinusoidal)")
    print("-" * 50)
    pos_enc = positional_encoding(seq_length, d_model)
    print(f"   Sequence length:         {seq_length}")
    print(f"   Model dimension:         {d_model}")
    print(f"   Output shape:            {pos_enc.shape}")

    # Test token embedding
    print("\n3. Token Embedding")
    print("-" * 50)
    key, subkey = random.split(key)
    token_emb = TokenEmbedding(vocab_size, d_model, subkey)
    key, subkey = random.split(key)
    token_ids = random.randint(subkey, (batch_size, seq_length), 0, vocab_size)
    embedded = token_emb(token_ids)
    print(f"   Vocabulary size:         {vocab_size}")
    print(f"   Token IDs shape:         {token_ids.shape}")
    print(f"   Embedded shape:          {embedded.shape}")

    # Test learnable positional encoding
    print("\n4. Learnable Positional Encoding")
    print("-" * 50)
    key, subkey = random.split(key)
    learnable_pos = LearnablePositionalEncoding(seq_length * 2, d_model, subkey)
    pos_output = learnable_pos(seq_length)
    print(f"   Max sequence length:     {seq_length * 2}")
    print(f"   Output shape:            {pos_output.shape}")

    # Test transformer embedding
    print("\n5. Transformer Embedding (Token + Positional)")
    print("-" * 50)
    key, subkey = random.split(key)
    transformer_emb = TransformerEmbedding(vocab_size, d_model, seq_length * 2, subkey, learnable_pos=True)
    emb_output = transformer_emb(token_ids)
    print(f"   Token IDs shape:         {token_ids.shape}")
    print(f"   Output shape:            {emb_output.shape}")

    # Test multi-head attention
    print("\n6. Multi-Head Attention")
    print("-" * 50)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha_output = mha(x, x, x)
    print(f"   Number of heads:         {num_heads}")
    print(f"   Input shape:             {x.shape}")
    print(f"   Output shape:            {mha_output.shape}")

    # Test feed-forward network
    print("\n7. Feed-Forward Network")
    print("-" * 50)
    ff = FeedForward(d_model=d_model, d_ff=d_ff)
    ff_output = ff(x)
    print(f"   Hidden dimension:        {d_ff}")
    print(f"   Input shape:             {x.shape}")
    print(f"   Output shape:            {ff_output.shape}")

    # Test encoder layer
    print("\n8. Encoder Layer")
    print("-" * 50)
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    layer_output = encoder_layer(x)
    print(f"   Input shape:             {x.shape}")
    print(f"   Output shape:            {layer_output.shape}")

    # Test full encoder
    print("\n9. Full Encoder")
    print("-" * 50)
    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    encoder_output = encoder(x)
    print(f"   Number of layers:        {num_layers}")
    print(f"   Input shape:             {x.shape}")
    print(f"   Output shape:            {encoder_output.shape}")

    # Test causal mask
    print("\n10. Causal Mask")
    print("-" * 50)
    causal_mask = create_causal_mask(seq_length)
    print(f"   Sequence length:         {seq_length}")
    print(f"   Mask shape:              {causal_mask.shape}")
    print(f"   Mask (5x5 corner):\n{causal_mask[:5, :5]}")

    # Test decoder layer
    print("\n11. Decoder Layer")
    print("-" * 50)
    key, subkey = random.split(key)
    decoder_layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, key=subkey)
    decoder_layer_output = decoder_layer(x, encoder_output, causal_mask)
    print(f"   Input shape:             {x.shape}")
    print(f"   Encoder output shape:    {encoder_output.shape}")
    print(f"   Output shape:            {decoder_layer_output.shape}")

    # Test full decoder
    print("\n12. Full Decoder")
    print("-" * 50)
    key, subkey = random.split(key)
    decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, key=subkey)
    decoder_output = decoder(x, encoder_output)
    print(f"   Number of layers:        {num_layers}")
    print(f"   Input shape:             {x.shape}")
    print(f"   Output shape:            {decoder_output.shape}")

    # Test output layer
    print("\n13. Output Layer")
    print("-" * 50)
    key, subkey = random.split(key)
    output_layer = OutputLayer(d_model, vocab_size, subkey)
    log_probs = output_layer(decoder_output)
    print(f"   Input shape:             {decoder_output.shape}")
    print(f"   Output shape:            {log_probs.shape}")
    print(f"   Sum of probs (should≈1): {jnp.exp(log_probs[0, 0]).sum():.6f}")

    # Test full Transformer (encoder-decoder)
    print("\n14. Full Transformer (Encoder-Decoder)")
    print("-" * 50)
    key, subkey = random.split(key)
    transformer = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=seq_length * 2,
        key=subkey,
    )
    key, subkey = random.split(key)
    src_tokens = random.randint(subkey, (batch_size, seq_length), 0, vocab_size)
    key, subkey = random.split(key)
    tgt_tokens = random.randint(subkey, (batch_size, seq_length), 0, vocab_size)
    transformer_output = transformer(src_tokens, tgt_tokens)
    print(f"   Source shape:            {src_tokens.shape}")
    print(f"   Target shape:            {tgt_tokens.shape}")
    print(f"   Output shape:            {transformer_output.shape}")

    # Test Transformer LM (decoder-only)
    print("\n15. Transformer Language Model (Decoder-Only)")
    print("-" * 50)
    key, subkey = random.split(key)
    lm = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=seq_length * 2,
        key=subkey,
    )
    lm_output = lm(token_ids)
    print(f"   Input shape:             {token_ids.shape}")
    print(f"   Output shape:            {lm_output.shape}")

    print("\n" + "=" * 60)
    print("All 15 components tested successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
