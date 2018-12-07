import torch.nn as nn
import torch
import math


class CoordinateEmbedding(nn.Module):
    """
        The CoordinateEmbedding Layer Pt
        P(pos,t,2j) = sin(pos/10000^(2j/dmodel))+sin(t/10000^(2j/dmodel))
        P(pos,t,2j+1) = cos(pos/10000^(2j/dmodel))+cos(t/10000^(2j/dmodel))
    """

    def __init__(self, dmodel, MAXLEN=90):
        super().__init__()
        self.dmodel = dmodel
        self.log_space = math.log(10000)*(-1)*torch.arange(0,
                                                           self.dmodel, 2) / self.dmodel

    def forward(self, x, timestep: int):
        pe = torch.zeros(MAXLEN, self.dmodel, dtype=torch.float)
        position = torch.arange(0, MAXLEN).unsqueeze(
            1).type(torch.float)  # (MAXLEN, 1)
        t_term = math.log(10000)*(-1)*torch.arange(0,
                                                   self.dmodel, 2)/self.dmodel
        t_term = torch.exp(t_term.type(torch.float)).unsqueeze(
            0).repeat(MAXLEN, 1)  # (MAXLEN, dmodel/2)
        pe[:, 0::2] = torch.sin(
            position*torch.exp(self.log_space.type(torch.float))) + torch.sin(timestep*t_term)
        pe[:, 1::2] = torch.cos(
            position*torch.exp(self.log_space.type(torch.float))) + torch.cos(timestep*t_term)
        # pe:(MAXLEN, dmodel)
        pe = pe.unsqueeze(0)
        # pe:(1, MAXLEN, dmodel)

        length = x.size(1)
        x = x*math.sqrt(self.dmodel)
        x = x + pe[:, :length, :]
        return x
