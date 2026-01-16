"""
Training example for JAXformers.

This example demonstrates how to train a small Transformer language model
on a simple copy task: the model learns to repeat the input sequence.
"""

import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from typing import Dict, Tuple
from functools import partial

from transformer import TransformerLM
from decoder import create_causal_mask


def cross_entropy_loss(log_probs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cross-entropy loss.

    Args:
        log_probs: Log probabilities of shape (batch_size, seq_length, vocab_size)
        targets: Target indices of shape (batch_size, seq_length)

    Returns:
        Scalar loss value
    """
    batch_size, seq_length, vocab_size = log_probs.shape
    # Flatten for indexing
    log_probs_flat = log_probs.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    # Get log probability of correct tokens
    correct_log_probs = log_probs_flat[jnp.arange(log_probs_flat.shape[0]), targets_flat]
    return -jnp.mean(correct_log_probs)


def get_params(model: TransformerLM) -> Dict:
    """Extract trainable parameters from model."""
    return {
        "embedding": {
            "token": model.embedding.token_embedding.embedding,
            "pos": model.embedding.pos_encoding.encoding if model.embedding.learnable_pos else None,
        },
        "decoder": {
            f"layer_{i}": {
                "ff_w1": layer.feed_forward.layer1,
                "ff_b1": layer.feed_forward.bias1,
                "ff_w2": layer.feed_forward.layer2,
                "ff_b2": layer.feed_forward.bias2,
            }
            for i, layer in enumerate(model.decoder.layers)
        },
        "output": {
            "weight": model.output_layer.weight,
            "bias": model.output_layer.bias,
        },
    }


def set_params(model: TransformerLM, params: Dict) -> TransformerLM:
    """Set model parameters from dict."""
    model.embedding.token_embedding.embedding = params["embedding"]["token"]
    if model.embedding.learnable_pos and params["embedding"]["pos"] is not None:
        model.embedding.pos_encoding.encoding = params["embedding"]["pos"]

    for i, layer in enumerate(model.decoder.layers):
        layer_params = params["decoder"][f"layer_{i}"]
        layer.feed_forward.layer1 = layer_params["ff_w1"]
        layer.feed_forward.bias1 = layer_params["ff_b1"]
        layer.feed_forward.layer2 = layer_params["ff_w2"]
        layer.feed_forward.bias2 = layer_params["ff_b2"]

    model.output_layer.weight = params["output"]["weight"]
    model.output_layer.bias = params["output"]["bias"]

    return model


def generate_copy_batch(
    key: random.PRNGKey, batch_size: int, seq_length: int, vocab_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a batch for the copy task.
    Input: random sequence
    Target: same sequence shifted by 1 (next token prediction)
    """
    # Generate random tokens (excluding 0 which we reserve for padding)
    tokens = random.randint(key, (batch_size, seq_length + 1), 1, vocab_size)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    return inputs, targets


def train_step(
    model: TransformerLM,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    learning_rate: float = 0.001,
) -> Tuple[TransformerLM, float]:
    """
    Perform a single training step.

    Args:
        model: The transformer model
        inputs: Input token indices
        targets: Target token indices
        learning_rate: Learning rate

    Returns:
        Updated model and loss value
    """

    def loss_fn(params):
        # Temporarily set params
        set_params(model, params)
        log_probs = model(inputs)
        return cross_entropy_loss(log_probs, targets)

    params = get_params(model)
    loss, grads = value_and_grad(loss_fn)(params)

    # Simple SGD update
    def update_param(param, grad):
        if param is None or grad is None:
            return param
        return param - learning_rate * grad

    def update_nested(params, grads):
        if isinstance(params, dict):
            return {k: update_nested(params[k], grads[k]) for k in params}
        return update_param(params, grads)

    new_params = update_nested(params, grads)
    model = set_params(model, new_params)

    return model, loss


def main():
    print("=" * 60)
    print("JAXformers Training Example: Copy Task")
    print("=" * 60)

    # Hyperparameters
    vocab_size = 32
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128
    max_seq_length = 32
    batch_size = 16
    seq_length = 16
    num_steps = 100
    learning_rate = 0.01

    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Layers: {num_layers}")
    print(f"  FFN dimension: {d_ff}")

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training steps: {num_steps}")

    # Initialize model
    key = random.PRNGKey(42)
    key, model_key = random.split(key)

    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout_rate=0.0,  # No dropout for this simple example
        learnable_pos=True,
        key=model_key,
    )

    print("\nTraining...")
    print("-" * 40)

    for step in range(num_steps):
        key, batch_key = random.split(key)
        inputs, targets = generate_copy_batch(batch_key, batch_size, seq_length, vocab_size)

        model, loss = train_step(model, inputs, targets, learning_rate)

        if step % 10 == 0:
            # Calculate accuracy
            log_probs = model(inputs)
            predictions = jnp.argmax(log_probs, axis=-1)
            accuracy = jnp.mean(predictions == targets)
            print(f"Step {step:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

    print("-" * 40)

    # Final evaluation
    print("\nFinal Evaluation:")
    key, eval_key = random.split(key)
    inputs, targets = generate_copy_batch(eval_key, batch_size, seq_length, vocab_size)
    log_probs = model(inputs)
    predictions = jnp.argmax(log_probs, axis=-1)
    accuracy = jnp.mean(predictions == targets)
    loss = cross_entropy_loss(log_probs, targets)

    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Show example prediction
    print("\nExample Prediction:")
    print(f"  Input:      {inputs[0, :10].tolist()}")
    print(f"  Target:     {targets[0, :10].tolist()}")
    print(f"  Prediction: {predictions[0, :10].tolist()}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
