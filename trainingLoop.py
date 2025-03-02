import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, DataCollatorWithPadding
from transformer import Transformer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk


# dataset
# https://huggingface.co/datasets/Skylion007/openwebtext/tree/main
dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)
#shuffled_dataset = dataset['train'].shuffle(seed=321)

# setup information
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
embed_dim = 256
num_head = 12
vocab_length = tokenizer.vocab_size
context_length = 512
epochs = 10

# tokenize data
# def tokenize_function(examples):
#     return tokenizer(examples['text'], truncation=True, max_length=1024)
# # tokenized data
# if __name__ == "__main__":    
#     tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc = 14)
#     tokenized_dataset.save_to_disk('C:/Users/adamt/OneDrive/Documents/llmCOOL')
dataset = load_from_disk("C:/Users/adamt/OneDrive/Documents/llmCOOL/train_512")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)

transformer = Transformer(embed_dim, num_head, vocab_length, context_length)


for epoch in range(epochs):
    print(f"epoch({epoch + 1}/{epochs})")
    # data["input_ids"] will be the input into the transformer
    for i, data in enumerate(dataloader, 0):
        out = transformer.forward(data["input_id"])
        # state: (batch_size, context_length. vocab_size)
        # need to compare it to (batch_size, context_length) where each element in context_length holds the ground truth