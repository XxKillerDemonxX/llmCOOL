import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTES TO SELF:
# add dropout layers wherever needed
# make sure everything is on the right device layer, ie. need device = device...


# -----MULTI HEAD ATTENTION LAYER-----
# embed_dim: embedding dimension of the input tokens
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        # head_dim should be an integer
        self.head_dim = embed_dim//num_heads
        self.weightQKV = nn.Linear(embed_dim, embed_dim*3, device=device)
        self.weightOut = nn.Linear(embed_dim, embed_dim, device=device)
    # matrix shape of x should follow (batch_size, context_length, embed_dim)
    # maybe add another parameter for data["attention_mask"]?
    def forward(self, x):
        batch_size, context_length, emb = x.shape
        # QKV (batch_size, context_length, embed_dim*3)
        QKV = self.weightQKV(x)
        # q (batch_size, context_length, embed_dim)
        # k (batch_size, context_length, embed_dim)
        # v (batch_size, context_length, embed_dim)
        q, k, v = QKV.split(self.embed_dim, dim = 2)
        # view as (batch_size, num_heads, context_length, head_dim)
        q = q.reshape(batch_size, self.num_heads, context_length, self.head_dim)
        k = k.reshape(batch_size, self.num_heads, context_length, self.head_dim)
        v = v.reshape(batch_size, self.num_heads, context_length, self.head_dim)

        # -> (batch_size, num_heads, context_length, context_length)
        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        # scaling
        scaling_factor = self.head_dim ** 0.5
        attn_score = attn_score/scaling_factor

        # masking
        tril = torch.tril((torch.ones(context_length, context_length, device=self.device)))
        # unsqueeze so (1, 1, context_length, context_length) to broadcast properly
        attn_score = attn_score.masked_fill(tril.unsqueeze(0).unsqueeze(0) == 0, float('-inf')).to(self.device)

        # softmax
        attn_weights = F.softmax(attn_score, dim = -1)
        attn_output = torch.matmul(attn_weights, v)
        # permute to (batch_size, context_length, num_heads, head_dim) (num_heads * head_dim = embed_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        # view to (batch_size, context_length, embed_dim)
        attn_output = attn_output.contiguous().view(batch_size, context_length, -1)

        return self.weightOut(attn_output)
    def param(self):
        return list(self.parameters())

#-----LAYER NORM-----
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, device):
        super(LayerNorm, self).__init__()
        self.weights = nn.Parameter(torch.ones(embed_dim)).to(device)
        self.bias = nn.Parameter(torch.zeros(embed_dim)).to(device)
    def forward(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        y = (x - xmean) / torch.sqrt(xvar)
        self.out = self.weights * y + self.bias
        return self.out
    def param(self):
        return list(self.parameters())
    
#-----MLP-----
class FeedForward(nn.Module):
    def __init__(self, embed_dim, device):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim, device=device),
        )
    def forward(self, x):
        return self.layers(x)
    def param(self):
        return list(self.parameters())
    
#-----EMBEDDING LAYER-----
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, device):
        super(EmbeddingLayer, self).__init__()
        #creates embedding with xavier intializtion so weights are drawn from uniform distribution, helps converge faster
        self.embedding = nn.init.xavier_uniform_(torch.empty(vocab_size, embed_dim)).to(device)
    #take input of shape (batch_size, context_length)
    def forward(self, x):
        #return output of shape (batch_size, context_length, embed_dim)
        return self.embedding[x]
    def param(self):
        return list(self.parameters())


