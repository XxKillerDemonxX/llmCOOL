import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, DataCollatorWithPadding
from transformer import Transformer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
embed_dim = 256
num_head = 8
vocab_length = tokenizer.vocab_size
context_length = 512
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
transformer = Transformer(embed_dim, num_head, vocab_length, context_length, device).to(device)
state_dict = torch.load('model_weights.pth')
transformer.load_state_dict(state_dict)

inputdata = "what is the capital of israel"

input = tokenizer(inputdata, return_tensors="pt")["input_ids"]
logits = transformer.forward(input)       # [B, T, V]

last_logits = logits[:, -1, :]        # [B, V]
last_logits = torch.clamp(last_logits, min=-100, max=100)
probs = torch.softmax(last_logits, dim=-1)  # convert to valid probability dist
#probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
probs = torch.clamp(probs, min=1e-9, max=1.0)
sample = torch.multinomial(probs, num_samples=1)
out = []
while sample != 50256:
    out.append(sample)
    logits = transformer.forward(torch.tensor(out).unsqueeze(0).to(device))
    last_logits = logits[:, -1, :]
    last_logits = torch.clamp(last_logits, min=-100, max=100)
    probs = torch.softmax(last_logits, dim=-1)
    #probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, min=1e-9, max=1.0)
    sample = torch.multinomial(probs, num_samples = 1)
print(tokenizer.decode(torch.tensor(out)))