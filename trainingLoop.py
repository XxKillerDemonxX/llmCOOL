import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from transformer import Transformer
from datasets import load_dataset


# dataset
# https://huggingface.co/datasets/Skylion007/openwebtext/tree/main
dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)
#shuffled_dataset = dataset['train'].shuffle(seed=321)

# setup information
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embed_dim = 256
num_head = 12
vocab_length = tokenizer.vocab_size
context_length = 512
epochs = 10

# tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)
# tokenized data
if __name__ == "__main__":    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc = 14)
    tokenized_dataset.save_to_disk('C:/Users/adamt/OneDrive/Documents/llmCOOL')
# for i in range(epochs):
#     print(tokenized_dataset['train'][i])