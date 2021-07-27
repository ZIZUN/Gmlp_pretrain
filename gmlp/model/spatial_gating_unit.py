import torch
from torch import nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, causal = False, act = nn.Identity()):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal

        self.norm = nn.LayerNorm(dim_out)

        self.dim_seq = dim_seq
        self.w_ = nn.Parameter(torch.zeros(dim_seq, dim_seq), requires_grad=True)
        self.b_ = nn.Parameter(torch.ones(dim_seq), requires_grad=True)

        self.act = act

    def forward(self, x, mask=None, gate_res = None): # x -> bsz, len, hidden*6
        device, n = x.device, x.shape[1]

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.w_, self.b_

        if self.causal:  # for input seq_len when finetuning
            weight, bias = weight[:n, :n], bias[:n]

        if mask is not None:
            weight = weight.masked_fill(mask == 0, 0)   # masking for padding token.

        gate = torch.matmul(weight, gate) + bias[None, :, None]   # WZ + b

        if gate_res is not None:
            gate = gate + gate_res

        return self.act(gate) * res