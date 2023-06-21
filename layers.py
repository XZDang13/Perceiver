import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, x):
        x = self.net(x)
        x = F.dropout(x, self.drop_prob)

        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head=4, drop_prob=0.1):
        super().__init__()
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.scale = self.head_dim ** -0.5
        self.drop_prob = drop_prob
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_head), [q, k, v])
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.drop_prob)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_linear(out)
        out = F.dropout(out, self.drop_prob)

        return out
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_head=8) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.self_attention = MultiHeadAttention(dim, num_head)
        self.feed_forward = FeedForward(dim)

    def forward(self, x, mask=None):
        x = self.norm_1(self.self_attention(x, x, x, mask) + x)
        x = self.norm_2(self.feed_forward(x) + x)

        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_head=8) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.self_attention = MultiHeadAttention(dim, num_head)
        self.feed_forward = FeedForward(dim)

    def forward(self, x, memory, mask=None):
        x = self.norm_1(self.self_attention(x, memory, memory, mask) + x)
        x = self.norm_2(self.feed_forward(x) + x)

        return x

class Attention(nn.Module):
    def __init__(self, latent_dim, num_self_attention) -> None:
        super().__init__()

        self.cross_attention = CrossAttention(latent_dim)
        
        self.self_attention_block = nn.ModuleList(
            [SelfAttention(latent_dim) for _ in range(num_self_attention)]
        )

    def forward(self, x, memory, cross_attention_mask=None, self_attention_mask=None):
        x = self.cross_attention(x, memory, cross_attention_mask)
        
        for block in self.self_attention_block:
            x = block(x, self_attention_mask)

        return x      