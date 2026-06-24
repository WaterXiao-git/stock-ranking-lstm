import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 10
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001

# ========== 1. 数据预处理 ==========
def preprocess(df, scaler=None, is_train=True):
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by=['股票代码', '日期'])

    # 构造涨跌幅
    df['涨跌幅'] = (df['收盘'] - df['开盘']) / df['开盘']

    # 对涨跌幅做标准化（训练阶段记录均值和方差）
    global mean, std
    if is_train:
        mean = df['涨跌幅'].mean()
        std = df['涨跌幅'].std()
    df['涨跌幅'] = (df['涨跌幅'] - mean) / (std + 1e-6)

    df['成交量'] = np.log1p(df['成交量'])
    df['成交额'] = np.log1p(df['成交额'])

    # 保留原始股票代码用于输出
    df['原始股票代码'] = df['股票代码']

    features = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']

    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if is_train:
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        return df, features, scaler
    else:
        df[features] = scaler.transform(df[features])
        return df, features

# ========== 2. 构建序列数据 ==========
def create_sequences(df, seq_len, features):
    X, y = [], []
    grouped = df.groupby('股票代码')
    for _, group in grouped:
        group = group.sort_values('日期')
        if len(group) <= seq_len:  # 👈 跳过样本不足的股票
            continue
        for i in range(len(group) - seq_len):
            seq = group.iloc[i:i + seq_len]
            label = group.iloc[i + seq_len]['涨跌幅']
            if not np.isnan(label):
                X.append(seq[features].values)
                y.append(label)
    return np.array(X), np.array(y)


# ========== 3. 模型定义 ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze()

# ========== 4. 模型训练 ==========
def train_model(model, X_train, y_train, X_val, y_val):
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            xb = torch.tensor(X_train[i:i + BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_train[i:i + BATCH_SIZE], dtype=torch.float32).to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE))
            val_loss = criterion(val_pred, torch.tensor(y_val, dtype=torch.float32).to(DEVICE))

        train_losses.append(total_loss / len(X_train))
        val_losses.append(val_loss.item())
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss.item():.6f}")

    # 可视化 Loss 曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# ========== 5. 预测（使用原始股票代码） ==========
def predict_latest(df, model, seq_len, features):
    grouped = df.groupby('股票代码')
    pred_list = []

    for code, group in grouped:
        group = group.sort_values('日期')
        if len(group) >= seq_len:
            seq = group.iloc[-seq_len:][features].values
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(seq_tensor).cpu().item()

            raw_code = group.iloc[-1]['原始股票代码']
            pred_list.append((raw_code, pred))

    pred_list_sorted = sorted(pred_list, key=lambda x: x[1], reverse=True)
    return pred_list_sorted

# ========== 6. 主流程 ==========
# 训练数据
train = pd.read_csv('train.csv')
train, features, scaler = preprocess(train, is_train=True)

train = pd.read_csv('train.csv')
print("训练数据总行数：", len(train))
print("股票代码数量：", train['股票代码'].nunique())

# 查看每只股票的数据量
counts = train['股票代码'].value_counts()
print("股票样本量分布（前10）：")
print(counts.head(10))
print("有足够长度的股票数量：", sum(counts > SEQ_LEN))

X, y = create_sequences(train, SEQ_LEN, features)
print(f"创建序列数据：X shape = {X.shape}, y shape = {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LSTMModel(input_size=len(features)).to(DEVICE)
print(f"✅ 使用设备: {DEVICE}")
train_model(model, X_train, y_train, X_val, y_val)

# 测试预测
test = pd.read_csv('test.csv')
test, _ = preprocess(test, scaler=scaler, is_train=False)
prediction_result = predict_latest(test, model, SEQ_LEN, features)

top10 = [x[0] for x in prediction_result[:10]]
bottom10 = [x[0] for x in prediction_result[-10:]]

result = pd.DataFrame({
    '涨幅最大股票代码': top10,
    '涨幅最小股票代码': bottom10
})
result.to_csv('result.csv', index=False, encoding='utf-8')
print("✅ 预测完成，结果保存在 result.csv")
