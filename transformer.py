import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from layers import MultiHeadAttention, LayerNorm, FeedForward, EmbeddingLayer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embed_dim = 768
num_head = 12
vocab_length = tokenizer.vocab_size
context_length = 1024

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
        self.embedding = EmbeddingLayer(vocab_length, embed_dim)
        # positional embedding: has an embedding for token position
        self.positionEmbedding = EmbeddingLayer(context_length, embed_dim)
        # transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(3)])
        # final linear layer to output of shape (batch_size, context_length, vocab_length)
        # when generating output, we would take the last index in context_length and use it's vocab_length to predict the next token
        self.outputLinear = nn.Linear(embed_dim, vocab_length)
    def forward(self, x):
        # get embedding data of x tokens
        x_emb = self.embedding.forward(x)
        # get positional embedding data of x tokens
        x_pos = self.positionEmbedding.forward(x)
        # add positional embedding to embedding
        x_out = x_emb + x_pos
        #
        for block in self.blocks:
            x_out = block.forward(x_out)

        self.out = self.outLinear(x_out)

        return self.out


