import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.control.control_plots import plt
from torch.utils.data import DataLoader
import numpy as np
from news_data import load_and_preprocess_data, NewsDataset, collate_fn


# --- 1. 替换为LSTM模型 ---
class OptimizedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=3, dropout=0.3):
        super(OptimizedLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 双向LSTM需要乘以2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # 拼接双向LSTM的最后一个隐藏状态
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(hidden)
        return output

# --- 2. 训练函数 ---
def train():
    print("开始加载和预处理数据...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = load_and_preprocess_data()

    # 将标签转为 tensor
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"训练集标签唯一值: {sorted(list(set(y_train.numpy())))}")
    print(f"验证集标签唯一值: {sorted(list(set(y_val.numpy())))}")
    print(f"测试集标签唯一值: {sorted(list(set(y_test.numpy())))}")

    print("创建数据加载器...")
    train_dataset = NewsDataset(X_train, y_train)
    val_dataset = NewsDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    print("初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedLSTMModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=128,
                      num_layers=2, dropout=0.5).to(device)

    # --- 优化器和损失函数 ---
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    print("开始训练...")
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")

        # --- 早停机制 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print(f"  -> New best model saved! Best Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    evaluate_model(model, device, val_loader, "Validation")
    test_dataset = NewsDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    evaluate_model(model, device, test_loader, "Test")


def evaluate_model(model, device, dataloader, name=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    train()