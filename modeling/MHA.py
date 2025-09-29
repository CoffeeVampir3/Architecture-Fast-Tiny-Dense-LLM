import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from einops import rearrange

class VarlenMHA(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_attn_heads
        assert self.hidden_size % self.n_heads == 0

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.w_o.weight)
        nn.init.normal_(self.w_q.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.w_k.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.006)

    def forward(self, x, cu_seqlens=None, max_seqlen=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Where s -> total sequence length (of all batches combined), h -> number attn heads, d -> hidden dimension
        q = rearrange(q, 's (h d) -> s h d', h=self.n_heads)
        k = rearrange(k, 's (h d) -> s h d', h=self.n_heads)
        v = rearrange(v, 's (h d) -> s h d', h=self.n_heads)

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True
        )

        out = rearrange(out, 's h d -> s (h d)')

        return self.w_o(out)
