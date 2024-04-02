# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
print(torch.__version__)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim,2):
                pe[pos, i] = math.sin(i / ((1e4**((2*i)/embed_dim))))
                pe[pos, i + 1] = math.cos(i / ((1e4**((2*(i+1))/embed_dim))))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim) 

    def forward(self, query, key, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)

        seq_length_query = query.size(1)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        q = self.query_matrix(query)   
        k = self.key_matrix(key)       # (32x10x8x64)
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention and adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        scores = F.softmax(product, dim=-1)
 
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor, n_heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim, embed_dim)
        )
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, query, key, value):
        attention_out = self.attention(query, key, value)
        attention_residual_out = attention_out + value
        norm1_out = self.drop1(self.norm1(attention_residual_out))
        feed_fwd_out = self.ff(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.drop2(self.norm2(feed_fwd_residual_out))
        return norm2_out


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = Embedding(vocab_size=vocab_size, embed_dim=embed_dim)
        self.positional_encoder = PositionalEncoding(max_seq_len=seq_len, embed_dim=embed_dim)

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out)
        return out
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim=embed_dim, expansion_factor=expansion_factor, n_heads=n_heads)

    def forward(self, key, query, x, mask):
        #we need to pass mask mask only to fst attention
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(key, query, value)

        
        return out
    

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEncoding(seq_len, embed_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) 
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        x = self.word_embedding(x)  #32x10x512
        x = self.position_embedding(x) #32x10x512
        x = self.dropout(x)
     
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask) 

        out = F.softmax(self.fc_out(x))
        return out
    

class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        src_vocab_size,
        target_vocab_size,
        seq_length,
        num_layers=2,
        expansion_factor=4, 
        n_heads=8,
    ):
        super(Transformer, self).__init__()

        self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(
            seq_len=seq_length, 
            vocab_size=src_vocab_size, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            expansion_factor=expansion_factor, 
            n_heads=n_heads,
        )

        self.decoder = TransformerDecoder(
            target_vocab_size=target_vocab_size, 
            embed_dim=embed_dim, 
            seq_len=seq_length, 
            num_layers=num_layers,
            expansion_factor=expansion_factor, 
            n_heads=n_heads,
        )
        
    
    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(
            torch.ones((trg_len, trg_len))
        ).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def decode(self,src,trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): #10
            out = self.decoder(out, enc_out, trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
   
        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs