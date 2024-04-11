import jax.numpy as jnp

class Embedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.embedding = jnp.zeros((vocab_size, d_model))

    def __call__(self, x: jnp.array) -> jnp.ndarray:
        return self_embedding[x]
    

# torch version

# class Embeddings(nn.Module):
#  def __init__(self, vocab_size: int, d_model: int):
    """
    Args:
      vocab_size:     size of vocabulary
      d_model:        dimension of embeddings
    """
    # inherit from nn.Module
#    super().__init__()   
     
    # embedding look-up table (lut)                          
#    self.lut = nn.Embedding(vocab_size, d_model)   

    # dimension of embeddings 
#    self.d_model = d_model                          

#  def forward(self, x: Tensor):
    """
    Args:
      x:              input Tensor (batch_size, seq_length)
      
    Returns:
                      embedding vector
    """
    # embeddings by constant sqrt(d_model)
#    return self.lut(x) * math.sqrt(self.d_model) 