import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.utils.data as data
from MultiheadAttention import MultiheadAttention, Feedforward, MoutipleBranchAttention, Feedforward_alpha
import math
from utils import clonelayers, create_mask


class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, MAX_LEN=90):
        super().__init__()
        self.dmodel = dmodel
        pe = torch.zeros(MAX_LEN, dmodel, dtype=torch.float)
        position = torch.arange(MAX_LEN).unsqueeze(1).type(torch.float)
        log_space = math.log(10000)*(-1)*torch.arange(0, dmodel, 2)/dmodel
        div_term = torch.exp(log_space.type(torch.float))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)

        x = x * math.sqrt(self.dmodel)
        x = x + self.pe[:, :length, :]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dmodel, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention(dmodel, heads)
        self.ffn = Feedforward(dmodel)
        self.layernorm1 = nn.LayerNorm(dmodel)
        self.layernorm2 = nn.LayerNorm(dmodel)
        self.drop_out1 = nn.Dropout(dropout)
        self.drop_out2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        xx = self.layernorm1(x)
        x = x + self.drop_out1(self.attn(xx, xx, xx, mask))

        xx = self.layernorm2(x)
        x = x + self.drop_out2(self.ffn(xx))
        return x


class EncoderLayer_(nn.Module):
    """
        Weighted Transformer Network Encoder Layer
    """

    def __init__(self, dmodel, heads, dropout=0.1):
        super().__init__()
        self.attn = MoutipleBranchAttention(dmodel, heads)
        self.ffn = Feedforward_alpha(dmodel, heads)
        self.layernorm1 = nn.LayerNorm(dmodel)
        self.layernorm2 = nn.LayerNorm(dmodel)
        self.drop_out1 = nn.Dropout(dropout)
        self.drop_out2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # shape of x:(bs, length, dmodel)
        xx = self.drop_out1(self.attn(xx))
        xx = self.drop_out2(self.ffn(xx))
        x = self.layernorm1(x + xx)
        return x


class DecoderLayer_(nn.Module):
    """
        Weighted Transformer Network Decoder layer
    """

    def __init__(self, dmodel, heads, dropout=0.1):
        self.masked_attn = MultiheadAttention(dmodel, heads)
        self.attn = MoutipleBranchAttention(dmodel, heads)
        self.ffn = Feedforward_alpha(dmodel, heads)
        self.layernorm1 = nn.LayerNorm(dmodel)
        self.layernorm2 = nn.LayerNorm(dmodel)
        self.layernorm3 = nn.LayerNorm(dmodel)
        self.drop_out1 = nn.Dropout(dropout)
        self.drop_out2 = nn.Dropout(dropout)
        self.drop_out3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        xx = self.layernorm1(x)
        x = x + self.drop_out1(self.masked_attn(xx))

        # shape of xx:(bs, length, dmodel)
        xx = self.drop_out2(self.attn(x))
        xx = self.drop_out3(self.ffn(xx))

        x = self.layernorm2(x+xx)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dmodel, heads, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiheadAttention(dmodel, heads)
        self.attn = MultiheadAttention(dmodel, heads)
        self.ffn = Feedforward(dmodel)
        self.layernorm1 = nn.LayerNorm(dmodel)
        self.layernorm2 = nn.LayerNorm(dmodel)
        self.layernorm3 = nn.LayerNorm(dmodel)
        self.drop_out1 = nn.Dropout(dropout)
        self.drop_out2 = nn.Dropout(dropout)
        self.drop_out3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        xx = self.layernorm1(x)
        x = x + self.drop_out1(self.masked_attn(xx, xx, xx, trg_mask))

        xx = self.layernorm2(x)
        x = x + self.drop_out2(self.attn(xx, e_outputs, e_outputs, src_mask))

        xx = self.layernorm3(x)
        x = x + self.drop_out3(self.ffn(xx))
        return x


class Encoder(nn.Module):
    def __init__(self, dmodel, vocab_size, heads, N, maxlen):
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, dmodel)
        self.position_encoding = PositionalEncoding(dmodel, maxlen)
        self.layers = clonelayers(EncoderLayer(dmodel, heads), N)
        self.final_norm = nn.LayerNorm(dmodel)

    def forward(self, x, mask=None):
        x = self.word_embedding(x)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class Encoder_(nn.Module):
    """
        Weighted Transformer Encoder
    """

    def __init__(self, dmodel, vocab_size, heads, N, maxlen):
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(self.vocab_size, self.dmodel)
        self.position_encoding = PositionalEncoding(self.dmodel, maxlen)
        self.layers = clonelayers(EncoderLayer_(self.dmodel, heads), N)
        self.final_norm = nn.LayerNorm(self.dmodel)

    def forward(self, x, mask=None):
        x = self.word_embedding(x)
        x = self.position_encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        x = self.final_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dmodel, vocab_size, heads, N, maxlen):
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.N = N
        self.word_embedding = nn.Embedding(vocab_size, dmodel)
        self.position_encoding = PositionalEncoding(dmodel, maxlen)
        self.layers = clonelayers(DecoderLayer(dmodel, heads), N)
        self.final_norm = nn.LayerNorm(dmodel)

    def forward(self, trg, e_outputs, src_mask, trg_mask=None):
        x = self.word_embedding(trg)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        x = self.final_norm(x)  # (batch_size, MAX_LEN, dmodel)
        return x


class Decoder_(nn.Module):
    """
        Weighted Transformer Decoder
    """

    def __init__(self, dmodel, vocab_size, heads, N, maxlen):
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.N = N
        self.word_embedding = nn.Embedding(self.vocab_size, self.dmodel)
        self.position_encoding = PositionalEncoding(self.dmodel, maxlen)
        self.layers = clonelayers(DecoderLayer(self.dmodel, heads), N)
        self.final_norm = nn.LayerNorm(self.dmodel)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x = self.word_embedding(x)
        x = self.position_encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, e_outputs, src_mask, trg_mask)
        x = self.final_norm(x)
        return x


class Transformer(nn.Module):
    """
        vanila Transformer NetWork
    """

    def __init__(self, dmodel, vocab_size, heads, N_encoder, N_decoder, src_maxlen, trg_maxlen):
        super().__init__()
        self.encoder = Encoder(dmodel, vocab_size, heads,
                               N_encoder, src_maxlen)
        self.decoder = Decoder(dmodel, vocab_size, heads,
                               N_decoder, trg_maxlen)
        self.output_layer = nn.Linear(dmodel, vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, src_mask, trg_mask)
        outputs = self.output_layer(d_outputs)
        return outputs


class WeightedTransformer(nn.Module):
    """
        Weighted Transformer NetWork
    """

    def __init__(self, dmodel, vocab_size, heads, N_encoder, N_decoder, src_maxlen, trg_maxlen):
        self.encoder = Encoder_(
            dmodel, vocab_size, heads, N_encoder, src_maxlen)
        self.decoder = Decoder_(
            dmodel, vocab_size, heads, N_decoder, trg_maxlen)
        self.output_layer = nn.Linear(dmodel, vocab_size)

    def forward(self, input, target, src_mask, trg_mask):
        encoder_output = self.encoder(input, src_mask)
        decoder_output = self.decoder(
            target, encoder_output, src_mask, trg_mask)
        output = self.output_layer(decoder_output)
        return output


class Transformer_Encoder(nn.Module):
    """
        Only the Transformer Encoder of vanila Transformer
    """

    def __init__(self, dmodel, vocab_size, heads, N, src_maxlen):
        super().__init__()
        self.encoder = Encoder(dmodel, vocab_size, heads, N, src_maxlen)
        self.output_layer = nn.Linear(dmodel, vocab_size)

    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        output = self.output_layer(e_outputs)
        return output
