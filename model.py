from sympy.printing.pretty.pretty_symbology import sup
import torch
import torch.nn as nn
import numpy as np




class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
       super()
