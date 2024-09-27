# data_preparation.py

import os
from cshogi import CSA
from collections import defaultdict
import torch
from torch.utils.data import Dataset

def parse_csa_files(directory):
    sequences = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            moves = []
            for line in lines:
                line = line.strip()
                if line.startswith('+') or line.startswith('-'):
                    moves.append(line)
            if moves:
                sequences.append(moves)
    return sequences

def build_vocab(sequences):
    token_counts = defaultdict(int)
    for seq in sequences:
        for move in seq:
            token_counts[move] += 1

    token2id = {token: idx for idx, token in enumerate(token_counts.keys())}
    id2token = {idx: token for token, idx in token2id.items()}
    return token2id, id2token

def tokenize_sequences(sequences, token2id):
    tokenized_sequences = []
    for seq in sequences:
        tokenized_seq = [token2id[move] for move in seq]
        tokenized_sequences.append(tokenized_seq)
    return tokenized_sequences

class ShogiDataset(Dataset):
    def __init__(self, sequences, seq_length):
        self.inputs = []
        self.targets = []
        for seq in sequences:
            if len(seq) < seq_length + 1:
                continue
            for i in range(len(seq) - seq_length):
                self.inputs.append(seq[i:i+seq_length])
                self.targets.append(seq[i+seq_length])
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)
