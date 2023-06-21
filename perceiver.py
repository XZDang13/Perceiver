import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from .layers import Attention

class PerceiverEncoder(nn.Module):
    def __init__(self, num_latents, latent_dim, num_self_attention, num_attention, max_length=1000) -> None:
        super().__init__()
        
        self.num_attention = num_attention

        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, latent_dim))
        
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.attention = Attention(latent_dim, num_self_attention)

    def forward(self, memory):
        batch_size, length, _ = memory.size()
        memory = memory + self.pos_embedding[:, :length]
        x = repeat(self.latents, 'n d -> b n d', b = batch_size)
        
        for _ in range(self.num_attention):
            x = self.attention(x, memory)
        
        return x
    
class PerceiverDecoder(nn.Module):
    def __init__(self, latent_dim, num_self_attention, num_attention, max_length, causal=False) -> None:
        super().__init__()
        self.num_attention = num_attention

        self.attention = Attention(latent_dim, num_self_attention)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, latent_dim))
        self.causal = causal
        
    def forward(self, x, memory):
        _, length, _ = x.size()
        x = x + self.pos_embedding[:, :length]
        mask = None
        if self.causal:
            ones = torch.ones((length, length))
            mask = torch.tril(ones).view(1, 1, length, length).to(x.device)
        
        for _ in range(self.num_attention):
            x = self.attention(x, memory, self_attention_mask = mask)        
           
        return x
    
class Embedding(nn.Module):
    def __init__(self, in_dim, latent_dim) -> None:
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.embedding(x)