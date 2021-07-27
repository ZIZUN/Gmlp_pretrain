import torch.nn as nn

class LM(nn.Module):
    """
    Language Model (Masked Language Model)
    """
    def __init__(self, model, vocab_size):
        """
        :param model: model which should be trained
        :param vocab_size: total vocab size for maslked_lm
        """
        super().__init__()
        self.model = model
        self.mask_lm = MaskedLanguageModel(self.model.hidden, vocab_size)

    def forward(self, x):
        x = self.model(x)
        return self.mask_lm(x)

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.Layernorm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        return self.softmax(self.linear(self.Layernorm(x)))
