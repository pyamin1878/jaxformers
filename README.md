# JAXformers

A lightweight, from-scratch implementation of the Transformer architecture using JAX.

![Transformer Architecture](images/image.png)

JAXformers provides clean, readable implementations suitable for research experimentation, prototyping novel architectures, and production use cases requiring customizable attention mechanisms. The library leverages JAX's automatic differentiation and XLA compilation for high-performance training and inference on CPU, GPU, and TPU.

## Installation

```bash
pip install jax jaxlib
```

For GPU support:
```bash
pip install jax[cuda12]
```

## Quick Start

```bash
git clone https://github.com/pyamin1878/jaxformers
cd jaxformers
python main.py
```

## Usage

### Encoder-Decoder Transformer (Seq2Seq)

```python
from jax import random
from transformer import Transformer

# Initialize model
key = random.PRNGKey(42)
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout_rate=0.1,
    key=key,
)

# Forward pass
src_tokens = ...  # (batch_size, src_seq_len)
tgt_tokens = ...  # (batch_size, tgt_seq_len)
log_probs = model(src_tokens, tgt_tokens)

# Generate (autoregressive decoding)
generated = model.generate(src_tokens, max_length=100, start_token=1, end_token=2)
```

### Decoder-Only Language Model

```python
from jax import random
from transformer import TransformerLM

# Initialize model
key = random.PRNGKey(42)
model = TransformerLM(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout_rate=0.1,
    learnable_pos=True,  # Use learnable positional encodings
    key=key,
)

# Forward pass (returns log probabilities)
tokens = ...  # (batch_size, seq_len)
log_probs = model(tokens)
```

### Training Example

```bash
python train_example.py
```

See `train_example.py` for a complete training loop demonstrating:
- Loss computation with cross-entropy
- Gradient computation with `jax.value_and_grad`
- Parameter updates with SGD

## Architecture

| Module | File | Description |
|--------|------|-------------|
| Scaled Dot-Product Attention | `attention.py` | Core attention with optional masking |
| Positional Encoding | `positional_encoding.py` | Sinusoidal position embeddings |
| Token Embedding | `embedding.py` | Vocabulary to dense vectors |
| Learnable Positional Encoding | `embedding.py` | Trainable position embeddings |
| Transformer Embedding | `embedding.py` | Combined token + positional embedding |
| Multi-Head Attention | `encoder.py`, `decoder.py` | Parallel attention heads |
| Feed-Forward Network | `encoder.py`, `decoder.py` | Two-layer MLP with ReLU |
| Encoder Layer | `encoder.py` | Self-attention + FFN with residuals |
| Encoder | `encoder.py` | Stack of encoder layers |
| Decoder Layer | `decoder.py` | Masked self-attn + cross-attn + FFN |
| Decoder | `decoder.py` | Stack of decoder layers with causal mask |
| Output Layer | `transformer.py` | Linear projection + softmax |
| Transformer | `transformer.py` | Full encoder-decoder model |
| TransformerLM | `transformer.py` | Decoder-only language model |

## Features

- Full encoder-decoder Transformer for seq2seq tasks
- Decoder-only Transformer for language modeling
- Scaled dot-product attention with optional masking
- Multi-head attention with configurable heads
- Sinusoidal and learnable positional encodings
- Token embeddings with Xavier initialization
- Causal masking for autoregressive decoding
- Layer normalization and dropout
- Autoregressive generation with temperature sampling
- Pure JAX - no Flax, Haiku, or other dependencies
- XLA compilation support

## Project Structure

```
jaxformers/
├── attention.py           # Scaled dot-product attention
├── positional_encoding.py # Sinusoidal positional encodings
├── embedding.py           # Token and positional embeddings
├── encoder.py             # Encoder layers and full encoder
├── decoder.py             # Decoder layers and full decoder
├── transformer.py         # Full Transformer and LM models
├── main.py                # Component tests
├── train_example.py       # Training loop example
└── notebooks/
    └── quickstart.ipynb   # JAX tutorial
```

## Resources

![Attention Mechanism](images/image-1.png)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyamin1878/jaxformers/blob/main/notebooks/quickstart.ipynb)

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [JAX Documentation](https://jax.readthedocs.io/)

## Acknowledgments

Based on the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) (Vaswani et al., 2017).

```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  year = {2018},
}
```

## License

MIT
