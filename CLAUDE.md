# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXformers is a lightweight, from-scratch implementation of the Transformer architecture using JAX. It provides clean, readable implementations suitable for research experimentation, prototyping novel architectures, and production use cases requiring customizable attention mechanisms. The library leverages JAX's automatic differentiation and XLA compilation for high-performance training and inference on CPU, GPU, and TPU.

## Commands

**Install dependencies:**
```bash
pip install jax jaxlib
```

**Run component tests:**
```bash
python main.py
```

**Run training example:**
```bash
python train_example.py
```

## Architecture

The codebase implements the complete Transformer architecture:

### Core Components
- `attention.py` - Scaled dot-product attention with optional masking
- `positional_encoding.py` - Sinusoidal positional encodings
- `embedding.py` - Token embeddings, learnable positional encodings, combined transformer embedding

### Encoder
- `encoder.py` - Multi-head attention, feed-forward network, encoder layers, and full encoder stack

### Decoder
- `decoder.py` - Decoder layers with masked self-attention and cross-attention, causal mask generation, full decoder stack

### Full Models
- `transformer.py` - Complete Transformer (encoder-decoder), TransformerLM (decoder-only), and output layer with softmax

## JAX Patterns

This codebase uses pure JAX without Flax or Haiku:

- `jax.numpy` (imported as `jnp`) for array operations
- Classes with `__call__` methods for module-like behavior
- Xavier/Glorot initialization for weights
- JAX PRNG system with explicit key passing for randomness
- Optional `key` parameter for dropout (None = inference mode, key = training mode)

Key JAX functions used:
- `jit` - JIT compilation for performance
- `grad` / `value_and_grad` - Automatic differentiation
- `vmap` - Auto-vectorization for batching
- `random.split` / `random.fold_in` - PRNG key management

## Model Variants

### Transformer (encoder-decoder)
For sequence-to-sequence tasks like translation:
```python
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, ...)
output = model(src_tokens, tgt_tokens)
```

### TransformerLM (decoder-only)
For language modeling / text generation:
```python
model = TransformerLM(vocab_size, d_model, num_heads, num_layers, ...)
output = model(tokens)
```

## Parameter Shapes

- Token embedding: `(vocab_size, d_model)`
- Positional encoding: `(max_seq_length, d_model)`
- Attention Q/K/V: Split from `(batch, seq, d_model)` to `(batch, heads, seq, d_k)`
- FFN layer 1: `(d_model, d_ff)`
- FFN layer 2: `(d_ff, d_model)`
- Output projection: `(d_model, vocab_size)`
