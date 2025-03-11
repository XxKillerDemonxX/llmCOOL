import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, DataCollatorWithPadding
from transformer import Transformer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk

ngpu = 1
# dataset
# https://huggingface.co/datasets/Skylion007/openwebtext/tree/main
#dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)
#shuffled_dataset = dataset['train'].shuffle(seed=321)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(torch.cuda.is_available())
print(device)
# setup information
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
embed_dim = 256
num_head = 8
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
dataset = load_from_disk("/home/ubuntu/test/llmCOOL/train_1024")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

transformer = Transformer(embed_dim, num_head, vocab_length, context_length, device).to(device)
num_params = sum(p.numel() for p in transformer.parameters())
print(f"Total parameters: {num_params}")
adam = torch.optim.Adam(num_params, lr = 0.001)

for epoch in range(epochs):
    print(f"epoch({epoch + 1}/{epochs})")
    # data["input_ids"] will be the input into the transformer
    for i, data in enumerate(dataloader, 0):
        #forward pass
        out = transformer.forward(data["input_ids"])
        ground_truth = torch.roll(data["input_ids"], -1, 1).to(device)
        ground_truth[:, -1].fill_(tokenizer.eos_token_id)
        out = out.permute(0, 2, 1)
        loss = F.cross_entropy(out, ground_truth)

        adam.zero_grad()

        #backward pass
        loss.backward()
        #update
        adam.step()

