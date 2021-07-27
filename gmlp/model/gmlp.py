import torch.nn as nn
from .embedding import gMLPEmbedding
from .gmlp_block import gMLPBlock
from gmlp.model.utils import dropout_layers

class gMLP(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, dropout=0.1, seq_len=512, attn_dim = None, causal=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: Gmlp model hidden size
        :param n_layers: numbers of Gmlp blocks(layers)
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers

        self.embedding = gMLPEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.gMLP_blocks = nn.ModuleList([gMLPBlock(dim=hidden, dim_ff=hidden * 6,
                                                                             attn_dim=attn_dim, causal=causal,
                                                                             seq_len=seq_len, act=nn.Identity()) for _  in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)  # masking matrix for padding tokens.

        x = self.embedding(x)  # embedding the indexed sequence to sequence of vectors

        self.gMLP_blocks = self.gMLP_blocks if not self.training else dropout_layers(self.gMLP_blocks, 1.0)

        for gMLPBlock in self.gMLP_blocks:
            x = x + self.norm(gMLPBlock.forward(x, mask))  # residula connection

        return x