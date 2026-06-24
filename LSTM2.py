import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

# ==================== 数据加载和预处理 ====================
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def preprocess(train_df, test_df):
    feature_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
    label_col = '涨跌幅'

    train_df = train_df.sort_values(['股票代码', '日期'])
    test_df = test_df.sort_values(['股票代码', '日期'])

    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, feature_cols, label_col, scaler

# ==================== 序列构造 ====================
def create_sequences(df, feature_cols, label_col, seq_len=SEQ_LEN, is_train=True):
    X, y = [], []
    grouped = df.groupby('股票代码')
    for _, group in grouped:
        group = group.reset_index(drop=True)
        if len(group) < seq_len + (1 if is_train else 0):
            continue

        if not all(col in group.columns for col in feature_cols):
            continue

        for i in range(len(group) - seq_len):
            try:
                seq_x = group.loc[i:i + seq_len - 1, feature_cols].values
                X.append(seq_x)

                if is_train:
                    label = group.loc[i + seq_len, label_col]
                    y.append(label)
            except Exception as e:
                print(f"跳过异常序列: {e}")
                continue
    if is_train:
        return np.array(X), np.array(y)
    else:
        return np.array(X), group.loc[:, ['股票代码']].iloc[-1]

# ==================== LSTM 回归模型 ====================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

# ==================== 模型训练函数 ====================
def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss/len(train_loader.dataset):.4f} - Val Loss: {val_loss/len(val_loader.dataset):.4f}")

# ==================== 主流程 ====================
def main():
    train_df, test_df = load_data("train_23-25.csv", "test_23-25.csv")
    train_df, test_df, features, label_col, scaler = preprocess(train_df, test_df)

    X, y = create_sequences(train_df, features, label_col, is_train=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=BATCH_SIZE)

    model = LSTMRegressor(input_size=len(features)).to(DEVICE)
    train_model(model, train_loader, val_loader)

    # === 推理预测 ===
    model.eval()
    grouped = test_df.groupby('股票代码')
    predictions = []
    for code, group in grouped:
        group = group.reset_index(drop=True)
        if len(group) < SEQ_LEN:
            continue
        seq = group.iloc[-SEQ_LEN:][features].values
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(seq_tensor).item()
        predictions.append((code, pred))

    # 排序并输出结果
    predictions.sort(key=lambda x: x[1], reverse=True)
    top10 = [x[0] for x in predictions[:10]]
    bottom10 = [x[0] for x in predictions[-10:]]

    result_df = pd.DataFrame({'涨幅最大股票代码': top10, '涨幅最小股票代码': bottom10})
    result_df.to_csv("result1.csv", index=False, encoding="utf-8")
    print("预测完成，结果保存为 result.csv")

if __name__ == "__main__":
    main()
