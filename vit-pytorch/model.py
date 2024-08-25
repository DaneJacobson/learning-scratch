import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm()
        self.attention = nn.MultiheadAttention()
        self.layer_norm2 = nn.LayerNorm()