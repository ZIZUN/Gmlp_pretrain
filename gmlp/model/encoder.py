import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder model ( for start token )
    """
    def __init__(self, model, vocab_size):
        super().__init__()
        self.model = model
        self.Layernorm = nn.LayerNorm(self.model.hidden)
        self.linear = nn.Linear(self.model.hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):# bsz, len, hidden
        x = self.model(x)

        #encoding = self.linear(self.Layernorm(x))
        encoding = x.transpose(0, 1)[0]

        return self.linear(encoding)  # (bsz, hidden)