## JAXformers: Learning Transformers Implementation with JAX

![alt text](images/image.png)

This project aims to provide a simple and educational implementation of the Transformer architecture using the JAX library. The purpose is to learn and understand how Transformers work by writing the code from scratch and leveraging the capabilities of JAX for numerical computing and machine learning.

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

### Learning Objectives

By exploring and understanding the code in this project, you can learn:

- How the scaled dot-product attention mechanism works.
- How multi-head attention is implemented and used in Transformers.
- The structure and components of a Transformer layer.
- How to use JAX for numerical computations and building neural networks.
- The benefits of using JAX, such as automatic differentiation and XLA compilation.

### Resources

![alt text](images/image-1.png)

For more information about Transformers and JAX, refer to the following resources:

[Andrej Karpathy's micrograd (neural networks from scratch)](https://www.youtube.com/watch?v=VMj-3S1tku0)

[Build a Transformer in JAX from scratch: how to write and train your own models](https://theaisummer.com/jax-transformer/)

[Attention is all you need: Discovering the Transformer paper](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)

[JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

Feel free to explore the code, experiment with different configurations, and extend the project to deepen your understanding of Transformers and JAX.

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
