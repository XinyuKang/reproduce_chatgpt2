import torch
import math
import torch.nn as nn
from config import GPTConfig as cfg

def createPositionalEmbMat(seq, n_emb):
    vec = 1 / (10000**(2 * torch.arange(n_emb//2) / n_emb))
    pos = torch.arange(seq).reshape(seq, -1)
    mat = pos * vec # (seq, n_emb/2)
    sin_mat = torch.sin(mat).view(seq, n_emb//2, -1)
    cos_mat = torch.cos(mat).view(seq, n_emb//2, -1)
    pe_mat = torch.stack([sin_mat, cos_mat], dim=2).flatten(1, 3)
    return pe_mat # (seq, n_emb)

class MultiHeadAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = torch.nn.Parameter(torch.randn(cfg.n_embd, cfg.n_embd))
        self.W_k = torch.nn.Parameter(torch.randn(cfg.n_embd, cfg.n_embd))
        self.W_v = torch.nn.Parameter(torch.randn(cfg.n_embd, cfg.n_embd))

    def forward(self, x):
        # (B , seq , n_embd) @ (n_embd , n_head * n_hid) = (B , seq , n_head * n_hid)
        # where n_head * n_hid  = n_embd
        B, seq, n_emb = x.shape
        
        d_k = n_emb // cfg.n_head
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        q = q.view(B, seq, cfg.n_head, -1).permute(0,2,1,3) # (B , n_head, seq, n_hid)
        k_t = k.view(B, seq, cfg.n_head, -1).permute(0,2,3,1) # (B , n_head, n_hid, seq)
        v = v.view(B, seq, cfg.n_head, -1).permute(0,2,1,3) # (B , n_head, seq, n_hid)
        # Add causal mask
        mask = torch.triu(torch.full((seq, seq), -float('inf')), diagonal=1)
        q_kt = q @ k_t + mask
        masked_score = nn.functional.softmax(q_kt * (1 / math.sqrt(d_k)), dim=3)  # (B , n_head, seq, seq)
        new_v = masked_score @ v # (B , n_head, seq, n_hid)
        new_v = new_v.permute(0,2,1,3).view(B, seq, -1)  # (B , seq, n_head * n_hid)
        return new_v  # (B , seq, n_emb)

class AttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_emb = cfg.n_emb
        self.self_attn = MultiHeadAttn()
        self.layer_norm_1 = nn.LayerNorm(n_emb)
        self.layer_norm_2 = nn.LayerNorm(n_emb)
        self.FFN = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb),
        )
        self.drop_out = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.drop_out(self.self_attn(self.layer_norm_1(x))) + x
        x = self.layer_norm_2(x)
        x = self.drop_out(self.FFN(x)) + x
        return x


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.n_embd) # (50256 * 768)
        self.pe_mat = createPositionalEmbMat(cfg.block_size, cfg.n_embd)
        attn_blocks_layers = [AttentionBlock() for _ in range(cfg.n_layer)]
        self.attn_blocks = nn.Sequential(*attn_blocks_layers)
        self.drop_out = nn.Dropout(p=cfg.dropout)

    def forward(self, x): # x: (B * seq)
        B, seq = x.shape
        # Embedding layer
        embeddings = self.embedding_layer(x) # (B * seq * n_embd)
        # Positional Encoding broadcast
        embeddings_with_pos = embeddings +  self.pe_mat[:seq, :]
        # Dropout
        atten_input = self.drop_out(embeddings_with_pos)
        # Self Attention
        atten_output = self.attn_blocks(atten_input)
