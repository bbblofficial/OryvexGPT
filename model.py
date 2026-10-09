"""
OryvexGPT: a decoder-only transformer (GPT) written from scratch in PyTorch.
Pre-LayerNorm blocks, causal self-attention, GELU MLP, tied embeddings,
and a KV cache so generation stays fast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256     # max context length in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


# name -> (n_layer, n_head, n_embd, block_size)
PRESETS = {
    "nano":  (4, 4, 256, 256),     # ~3-4M params  - trains in minutes on a CPU
    "small": (6, 6, 384, 256),     # ~11M params   - good CPU/GPU balance
    "base":  (8, 8, 512, 512),     # ~27M params   - wants a GPU
    "large": (12, 12, 768, 512),   # ~87M params   - needs a decent GPU
}


def preset_config(name: str, vocab_size: int, block_size: int | None = None, dropout: float = 0.1):
    layers, heads, emb, block = PRESETS[name]
    return GPTConfig(vocab_size=vocab_size, block_size=block_size or block,
                     n_layer=layers, n_head=heads, n_embd=emb, dropout=dropout)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None, use_cache=False):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)
        if past is not None:
            assert T == 1, "cached decoding processes one token at a time"
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=past is None)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), present


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, past=None, use_cache=False):
        a, present = self.attn(self.ln1(x), past, use_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class OryvexGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init of residual projections
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None, past=None, use_cache=False):
        B, T = idx.shape
        offset = 0 if past is None else past[0][0].shape[2]
        if offset + T > self.cfg.block_size:
            raise ValueError(f"sequence of {offset + T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past is None else past[i], use_cache)
            presents.append(present)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss, (presents if use_cache else None)
