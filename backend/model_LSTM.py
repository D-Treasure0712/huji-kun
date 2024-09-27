# model_LSTM.py

import torch
import torch.nn as nn

class ShogiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_layers, dropout=0.1):
        super(ShogiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length)
        embed = self.dropout(self.embedding(x))  # (batch_size, seq_length, embed_size)
        output, (hn, cn) = self.lstm(embed)     # output: (batch_size, seq_length, hidden_dim)
        output = self.fc(output)                # (batch_size, seq_length, vocab_size)
        return output
