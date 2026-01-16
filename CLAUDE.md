# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXformers is a lightweight, from-scratch implementation of the Transformer architecture using JAX. It provides clean, readable implementations suitable for research experimentation, prototyping novel architectures, and production use cases requiring customizable attention mechanisms. The library leverages JAX's automatic differentiation and XLA compilation for high-performance training and inference on CPU, GPU, and TPU.

## Commands

**Install dependencies:**
```bash
pip install jax jaxlib
```

**Run the main script:**
```bash
python main.py
```

## Architecture

The codebase implements the standard Transformer architecture with the following components:

- **Attention mechanism**: Scaled dot-product attention (expects `attention.py`)
- **Positional encoding**: Sinusoidal position embeddings (expects `positional_encoding.py`)
- **Encoder**: Stack of encoder layers with multi-head attention, feed-forward networks, and layer normalization (`encoder.py`)

Note: The repository currently only contains `encoder.py`. The imports reference `attention.py` and `positional_encoding.py` which need to be implemented.

## JAX Patterns

This codebase uses JAX idioms:
- `jax.numpy` (imported as `jnp`) for array operations
- Classes with `__call__` methods for module-like behavior (not using Flax/Haiku)
- Manual weight initialization with `jnp.zeros`

Key JAX functions to be familiar with:
- `jit` - JIT compilation for performance
- `grad` - Automatic differentiation
- `vmap` - Auto-vectorization for batching

## Current TODO (from README)

- Add embedding layer for input token conversion
- Implement encoder and decoder modules
- Add final linear layer and softmax for output
- Add layer normalization after each sublayer
- Apply dropout regularization
- Optimize with JAX features
