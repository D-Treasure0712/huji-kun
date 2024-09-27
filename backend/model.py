# model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ShogiTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(ShogiTransformerModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(self.embed_size)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer(src_emb.transpose(0, 1), tgt_emb.transpose(0, 1))
        output = self.fc_out(output)
        return output.transpose(0, 1)

#強化学習

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShogiModel(nn.Module):
    def __init__(self, action_size):
        super(ShogiModel, self).__init__()
        self.conv1 = nn.Conv2d(33, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(128 * 9 * 9, action_size)
        self.fc_value = nn.Linear(128 * 9 * 9, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        policy = self.fc_policy(x)
        value = self.fc_value(x)
        return policy, value
