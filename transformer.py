import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, LayerNorm, FeedForward


embed_dim = 64
num_head = 8


class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_head)
        self.ff = FeedForward(embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.ff.forward(self.ln2.forward(x))
        return x




