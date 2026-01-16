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

```python
import jax.numpy as jnp
from jax import random
from encoder import Encoder

# Configuration
batch_size = 2
seq_length = 10
d_model = 64
num_heads = 8
d_ff = 256
num_layers = 2

# Create encoder
encoder = Encoder(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout_rate=0.1
)

# Generate input
key = random.PRNGKey(42)
x = random.normal(key, (batch_size, seq_length, d_model))

# Forward pass (inference mode - no dropout)
output = encoder(x)

# Forward pass (training mode - with dropout)
output = encoder(x, key=key)
```

## Architecture

| Module | File | Description |
|--------|------|-------------|
| Scaled Dot-Product Attention | `attention.py` | Core attention mechanism with optional masking |
| Positional Encoding | `positional_encoding.py` | Sinusoidal position embeddings |
| Multi-Head Attention | `encoder.py` | Parallel attention heads with head splitting |
| Feed-Forward Network | `encoder.py` | Two-layer MLP with ReLU activation |
| Encoder Layer | `encoder.py` | Self-attention + FFN with residual connections and layer norm |
| Encoder | `encoder.py` | Stack of encoder layers with positional encoding |

## Features

- Scaled dot-product attention with optional masking
- Multi-head attention with configurable number of heads
- Sinusoidal positional encodings
- Layer normalization
- Dropout regularization (training mode)
- Pure JAX - no Flax, Haiku, or other framework dependencies
- XLA compilation support for accelerated performance

## Project Structure

```
jaxformers/
├── attention.py           # Scaled dot-product attention
├── positional_encoding.py # Sinusoidal positional encodings
├── encoder.py             # Multi-head attention, FFN, encoder layers
├── main.py                # Demo script
└── notebooks/
    └── quickstart.ipynb   # JAX tutorial notebook
```

## Roadmap

- [ ] Token embedding layer
- [ ] Decoder module with cross-attention
- [ ] Final linear layer and softmax for output prediction
- [ ] Learnable positional encodings
- [ ] Training loop example

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
