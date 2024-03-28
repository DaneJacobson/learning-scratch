import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        out = self.embed(x)
        return out
    
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_model_dim
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        for pos in range(self.max_seq_len):
             for i in range(self.embed_dim):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        pe.register_buffer('pe', )
