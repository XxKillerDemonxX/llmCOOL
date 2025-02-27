import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from transformer import Transformer



# setup information
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embed_dim = 768
num_head = 12
vocab_length = tokenizer.vocab_size
context_length = 1024
epochs = 50

for i in range(epochs):
    