
from torch import nn
from .attention import Tiny_Attention
from .spatial_gating_unit import SpatialGatingUnit

class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        attn_dim = None,
        causal = False,
        act = nn.Identity(),
        dropout = 0.1
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Tiny_Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None
        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal,  act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x, mask=None):
        gate_res = self.attn(x, mask) if exists(self.attn) else None  # if you want tiny_attention

        x = self.proj_in(x)
        x = self.sgu(x, mask, gate_res = gate_res)
        x = self.proj_out(x)

        return x

def exists(val):
    return val is not None