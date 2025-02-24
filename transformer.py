import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, LayerNorm, FeedForward, EmbeddingLayer


embed_dim = 64
num_head = 8
vocab_length = 100
context_length = 100

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_head)
        self.ff = FeedForward(embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.ff.forward(self.ln2.forward(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # embedding: has an embedding for each token
        embedding = EmbeddingLayer(vocab_length, embed_dim)
        # positional embedding: has an embedding for token position
        positionEmbedding = EmbeddingLayer(context_length, embed_dim)
        # transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(3)])
        # final linear layer to output of shape (batch_size, context_length, vocab_length)
        # when generating output, we would take the last index in context_length and use it's vocab_length to predict the next token
        outputLinear = nn.Linear(embed_dim, vocab_length)
    def forward(self, x):
        return


