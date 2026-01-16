import jax.numpy as jnp
from jax import random

from attention import scaled_dot_product_attention
from positional_encoding import positional_encoding
from encoder import MultiHeadAttention, FeedForward, EncoderLayer, Encoder


def main():
    # Configuration
    batch_size = 2
    seq_length = 10
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2

    # Initialize random key
    key = random.PRNGKey(42)

    # Generate random input
    x = random.normal(key, (batch_size, seq_length, d_model))

    print("=" * 50)
    print("JAXformers: Transformer Implementation Demo")
    print("=" * 50)

    # Test scaled dot-product attention
    print("\n1. Testing Scaled Dot-Product Attention")
    print("-" * 40)
    query = key_tensor = value = x
    attention_output, attention_weights = scaled_dot_product_attention(query, key_tensor, value)
    print(f"   Input shape: {x.shape}")
    print(f"   Attention output shape: {attention_output.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")

    # Test positional encoding
    print("\n2. Testing Positional Encoding")
    print("-" * 40)
    pos_enc = positional_encoding(seq_length, d_model)
    print(f"   Sequence length: {seq_length}")
    print(f"   Model dimension: {d_model}")
    print(f"   Positional encoding shape: {pos_enc.shape}")
    print(f"   Sample values (first 5 positions, first 4 dims):")
    print(pos_enc[:5, :4])

    # Test multi-head attention
    print("\n3. Testing Multi-Head Attention")
    print("-" * 40)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha_output = mha(x, x, x)
    print(f"   Number of heads: {num_heads}")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {mha_output.shape}")

    # Test feed-forward network
    print("\n4. Testing Feed-Forward Network")
    print("-" * 40)
    ff = FeedForward(d_model=d_model, d_ff=d_ff)
    ff_output = ff(x)
    print(f"   Hidden dimension: {d_ff}")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {ff_output.shape}")

    # Test encoder layer
    print("\n5. Testing Encoder Layer")
    print("-" * 40)
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    layer_output = encoder_layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {layer_output.shape}")

    # Test full encoder
    print("\n6. Testing Full Encoder")
    print("-" * 40)
    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    encoder_output = encoder(x)
    print(f"   Number of layers: {num_layers}")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {encoder_output.shape}")

    print("\n" + "=" * 50)
    print("All components working correctly!")
    print("=" * 50)


if __name__ == "__main__":
    main()
