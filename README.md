## JAXformers: Transformer Implementation with JAX

![alt text](images/image.png)

JAXformers is a lightweight, from-scratch implementation of the Transformer architecture using JAX. It provides clean, readable implementations suitable for research experimentation, prototyping novel architectures, and production use cases requiring customizable attention mechanisms. The library leverages JAX's automatic differentiation and XLA compilation for high-performance training and inference on CPU, GPU, and TPU.

### Project Structure 

The project is organized into the following files:

`attention.py`: Contains the implementation of the scaled dot-product attention mechanism.

`transformer_layer.py`: Defines the multi-head attention and the complete Transformer layer.

`main.py`:  Demonstrates how to use the Transformer layer and perform a simple forward pass.

### Getting Started 

To run the code and experiment with the Transformer layer, follow these steps:

1. Install dependencies 

```python
pip install jax jaxlib
```

2. Clone the repo:

```
git clone https://github.com/pyamin1878/jaxformers
``` 

3. Navigate to the project directory:

```
cd jaxformers
```

4. Run the main script:

```python
python main.py 
```

### Features

- Scaled dot-product attention mechanism
- Multi-head attention with configurable heads
- Complete Transformer layer implementation
- Pure JAX with no additional framework dependencies
- XLA compilation support for accelerated performance
- Automatic differentiation for training

### Resources

![alt text](images/image-1.png)

For more information about Transformers and JAX, refer to the following resources:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyamin1878/jaxformers/blob/main/notebooks/quickstart.ipynb)


[Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

[Build a Transformer in JAX from scratch: how to write and train your own models](https://theaisummer.com/jax-transformer/)

[Attention is all you need: Discovering the Transformer paper](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)

Feel free to explore the code, experiment with different configurations, and extend the project for your research or production needs.

### TODO
```[tasklist]
- [ ] Add an embedding layer to convert input tokens into dense vectors
- [ ] Implement the encoder and decoder modules
- [x] Include positional encoding to incorporate positional information
- [ ] Implement the final linear layer and softmax for output prediction
- [ ] Add layer normalization after each sublayer 
- [ ] Apply dropout regularization to prevent overfitting
- [ ] Optimize performance and leverage JAX features
```

### Acknowledgments

This project is inspired by the Transformers architecture proposed in the paper ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) and is implemented using the [JAX](https://github.com/google/jax) library developed by Google.

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```
