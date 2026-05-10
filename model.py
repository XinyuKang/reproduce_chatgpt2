import torch
import torch.nn as nn
from config import GPTConfig as cfg
    

class GPT2(nn.module):
    def __init__(self):
        self.embedding_layer = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.n_embd) # (50256 * 1024)

    def forward(self, x): # x: (B * seq)
        # Embedding layer
        embeddings = self.embedding_layer(x) # (B * seq * n_embd)
        # Positional Encoding