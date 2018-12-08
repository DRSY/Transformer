import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from typing import Callable


def attention(query, key, value, mask, dropout: Callable):
    """
        the shape of qeury/key/value are both: (batch_size, heads, MAXLEN, d_k)
        implement the scaled dot product attention: A = softmax(query.dot(key.T)).dot(value)
        returns values that are of the same shape:(batch_size, heads, MAXLEN, d_k)
    """
    dim = query.size(-1)
    attn_score = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(dim)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        attn_score = attn_score.masked_fill(mask, -1e9)
    attn_score = F.softmax(attn_score, dim=-1)
    if dropout:
        attn_score = dropout(attn_score)
    return torch.matmul(attn_score, value), attn_score


class MultiheadAttention(nn.Module):
    """
        implement scaled dot product attention operation in number of heads parallely
    """

    def __init__(self, dmodel, heads, dropout=0.1):
        super().__init__()
        assert dmodel % heads == 0, "dimension not match"
        self.dmodel = dmodel
        self.heads = heads
        self.dk = dmodel // heads
        self.linears = utils.clonelayers(nn.Linear(dmodel, dmodel), 4)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        bs = query.size(0)
        query, key, value = [linear(x).view(bs, -1, self.heads, self.dk).transpose(
            1, 2) for x, linear in zip([query, key, value], self.linears)]
        outputs, _ = attention(query, key, value, mask, self.drop_out)
        outputs = outputs.transpose(1, 2).reshape(bs, -1, self.heads*self.dk)
        outputs = self.linears[-1](outputs)
        return outputs


class Feedforward(nn.Module):
    """
        Simple two layer FFN
    """

    def __init__(self, dmodel, dropout=0.1):
        super().__init__()
        self.dff = 2048
        self.l1 = nn.Linear(dmodel, self.dff)
        self.l2 = nn.Linear(self.dff, dmodel)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, input):
        x = self.drop_out(F.relu(self.l1(input)))
        x = self.l2(x)
        return x


class MoutipleBranchAttention(nn.Module):
    """
        The attention layer described in Weighted Transformer Network
    """

    def __init__(self, dmodel, heads, dropout=0.2):
        self.dmodel = dmodel
        self.heads = heads
        assert dmodel % heads == 0, "dimension not match"
        self.dk = self.dmodel / self.heads
        self.linears = utils.clonelayers(nn.Linear(dmodel, dmodel), 3)
        self.wo = nn.Linear(self.dk, self.dmodel)
        self.drop_out = nn.Dropout(dropout)
        self.k = nn.Parameter(torch.randn(self.heads, dtype=torch.float).unsqueeze(
            0).unsqueeze(-1).unsqueeze(-1))

    def forward(self, query, key, value, mask=None):
        bs = query.size(0)
        query, key, value = [l(x).view(bs, -1, self.heads, self.dk).transpose(1, 2)
                             for x, l in zip([query, key, value], self.linears)]
        # shape:(bs, heads, length, dk)
        outputs, _ = attention(query, key, value, mask, self.drop_out)

        # shape:(bs, heads, length, dmodel)
        outputs = self.wo(outputs)
        # shape:(bs, heads, length, dmodel)
        outputs = outputs * self.k
        return outputs


class Feedforward_alpha(nn.Module):
    """
        The FFN layer described in Weighted Transformer Network
    """

    dff = 2048

    def __init__(self, dmodel, heads, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.dmodel = dmodel
        self.l1 = nn.Linear(self.dmodel, Feedforward_alpha.dff)
        self.l2 = nn.Linear(Feedforward_alpha.dff, self.dmodel)
        self.drop_out = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.randn(
            self.heads, dtype=torch.float).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        # shape of x:(bs, heads, length, dmodel)
        x = self.drop_out(F.relu(self.l1(x)))
        x = self.l2(x)

        # shape:(bs, length, dmodel)
        x = x * self.alpha
        x = torch.sum(x, 1)
        return x
