# train_LSTM.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_preparation import parse_csa_files, build_vocab, tokenize_sequences, ShogiDataset
from model_LSTM import ShogiLSTMModel  # LSTM モデルをインポート
import os

# 1. データの準備（変更なし）
data_directory = 'kifu_data_fuji'  # 棋譜データのディレクトリ
sequences = parse_csa_files(data_directory)
token2id, id2token = build_vocab(sequences)
tokenized_sequences = tokenize_sequences(sequences, token2id)

# 2. ハイパーパラメータの設定
vocab_size = len(token2id)
embed_size = 256
hidden_dim = 512
num_layers = 2
seq_length = 10
batch_size = 32
num_epochs = 50
learning_rate = 1e-4

# 3. データセットの作成（変更なし）
dataset = ShogiDataset(tokenized_sequences, seq_length)

# データセットの分割（80%を学習データ、20%をテストデータ）
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 4. モデルの初期化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ShogiLSTMModel(vocab_size, embed_size, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 学習ループ
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for src_batch, tgt_batch in train_loader:
        src_batch = src_batch.to(device)  # (batch_size, seq_length)
        tgt_batch = tgt_batch.to(device)  # (batch_size)
        optimizer.zero_grad()

        output = model(src_batch)  # (batch_size, seq_length, vocab_size)
        output = output[:, -1, :]  # (batch_size, vocab_size)
        loss = criterion(output, tgt_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 精度の計算
        _, predicted = torch.max(output.data, 1)
        total += tgt_batch.size(0)
        correct += (predicted == tgt_batch).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    # テストデータで評価
    model.eval()
    total_loss_test = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for src_batch, tgt_batch in test_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            output = model(src_batch)
            output = output[:, -1, :]
            loss = criterion(output, tgt_batch)
            total_loss_test += loss.item()

            # 精度の計算
            _, predicted = torch.max(output.data, 1)
            total_test += tgt_batch.size(0)
            correct_test += (predicted == tgt_batch).sum().item()

    avg_loss_test = total_loss_test / len(test_loader)
    accuracy_test = 100 * correct_test / total_test

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%\n")

# 6. モデルの保存（変更なし）
if not os.path.exists('model'):
    os.makedirs('model')
torch.save(model.state_dict(), 'model/shogi_lstm_model.pth')

# トークン辞書の保存（変更なし）
import pickle
with open('model/token2id.pkl', 'wb') as f:
    pickle.dump(token2id, f)
with open('model/id2token.pkl', 'wb') as f:
    pickle.dump(id2token, f)
