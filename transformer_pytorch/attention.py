# attention mechanism

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_size, heads):

        # initializes parent class.
        super(Attention, self).__init__() 
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "the embed size needs to be divisible by heads"

        # Linear layer through which, parameters Keys (k), Queries (q) & Values (v) will be passed along.
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.foutput = nn.Linear(heads * self.head_dim, embed_size)

        def forward(self, values, keys, query, mask):
            N = query.shape[0]

            value_l, key_l, query_l = values.shape[1], keys.shape[1], query.shape[1]

            values = values.reshape(N, value_l, self.heads, self.head_dim)
            keys = keys.reshape(N, key_l, self.heads, self.head_dim)
            queries = query.reshape(N, query_l, self.heads, self.head_dim)

            energy = torch.einsum("nqhd,nkhd-->nhqk", [queries, keys])

            if mask is not None:
                # float("-1e20") is a very small number
                energy = energy.masked_fill(mask == 0, float("-1e20"))
            
            # attention formula Attention(Q,K,V) = softmax(QK^T/ sqrt(Dk))V
            attention = torch.softmax( energy / (self.embed_size ** (1/2)), dim=3)

            out = torch.einsum("nhql,nlhd-->nqhd", [attention, values]).reshape(
                N, query_l, self.heads*self.head_dim
            )

            out = self.foutput(out)
            return out
    
class TransformerBlck(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        
        # calling the super class
        super(TransformerBlck, self).__init__()
        self.attention = Attention(embed_size, heads)

        # normalization, LayerNorm & BatchNorm are similar
                        # BatchNorm takes average across batch and normalize 
                        # LayerNorm takes average for every example and normalize 

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.ff(x)

        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_l
    ):
        
        # calling the super class
        super(Encoder, self).__init__()
        
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(src_vocab, embed_size)
        self.positional_embed = nn.Embedding(max_l, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlck(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embed(x) + self.positional_embed(positions))

        for layer in self.layers:
            out = layer(out,out,out,mask)

        return out

class DecoderBlck(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlck, self).__init__()

        self.attention = Attention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.TransformerBlck_block = TransformerBlck(
            embed_size,heads,dropout,forward_expansion
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, tg_mask):
        attention = self.attention(x,x,x,tg_mask)
        query = self.dropout(self.norm(attention + x))

        out = self.TransformerBlck_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, tg_vocab_size, num_layers, embed_size, heads, forward_expansion, dropout, device, max_l):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embed = nn.Embedding(tg_vocab_size, embed_size)
        self.positional_embed = nn.Embedding(max_l, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlck(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
        )

        self.foutput = nn.Linear(embed_size, tg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embed(x) + self.positional_embed(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tg_mask)

        out = self.foutput(x)
        return out
    
class Transformer(nn.Module):
    def __init__(self,
            src_vocab_size,
            tg_vocab_size,
            src_pad_idx,
            tg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_l=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_l
        )

        self.decoder = Decoder(
            tg_vocab_size,
            embed_size,
            num_layers,
            forward_expansion,
            heads,
            dropout,
            device,
            max_l
        )

        self.src_pad_idx = src_pad_idx
        self.tg_pad_idx = tg_pad_idx
        self.device = device

    def make_src_mask(self,src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)
    
    def make_tg_mask(self, tg):
        N, tg_len = tg.shape
        tg_mask = torch.tril(torch.ones((tg_len,tg_len))).expand(
            N,1,tg_len,tg_len
        )

        return tg_mask.to(self.device)

    def forward(self, src, tg):
        src_mask = self.make_src_mask(src)
        tg_mask = self.make_tg_mask(tg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tg, enc_src, src_mask, tg_mask)
        return out