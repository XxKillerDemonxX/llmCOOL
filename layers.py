import torch
import torch.nn as nn
import torch.nn.functional as F



# -----MULTI HEAD ATTENTION LAYER-----
# embed_dim: embedding dimension of the input tokens
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # head_dim should be an integer
        self.head_dim = embed_dim//num_heads
        self.weightQKV = nn.Linear(embed_dim, embed_dim*3)
        self.weightOut = nn.Linear(embed_dim, embed_dim)
    # matrix shape of x should follow (batch_size, context_length, embed_dim)
    def forward(self, x):
        batch_size, context_length, emb = x.shape
        QKV = self.weightQKV(x)
        q, k, v = QKV.split(self.embed_dim, dim = 2)

        # view as (batch_size, num_heads, context_length, head_dim)
        q = q.view(batch_size, self.num_heads, context_length, self.head_dim)
        k = k.view(batch_size, self.num_heads, context_length, self.head_dim)
        v = v.view(batch_size, self.num_heads, context_length, self.head_dim)

        # -> (batch_size, num_heads, context_length, context_length)
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        # scaling
        scaling_factor = q.size(self.head_dim) ** 0.5
        attn_score = attn_score/scaling_factor

        # softmax
        attn_weights = F.softmax(attn_score, dim = -1)

        attn_output = torch.matmul(attn_weights, v)
        # permute to (batch_size, context_length, num_heads, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        # view to (batch_size, context_length, emb_size)
        attn_output = attn_output.contiguous.view(batch_size, context_length, -1)

        return self.weightOut(attn_output)
    def param(self):
        return list(self.parameters())


        