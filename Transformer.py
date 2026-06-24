import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from ta.momentum import RSIIndicator
from ta.trend import MACD

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 10
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ===== 1. 技术指标增强 =====
def add_technical_indicators(df):
    df_list = []
    for _, group in df.groupby('股票代码'):
        group = group.copy()
        # 添加 RSI
        rsi = RSIIndicator(close=group['收盘'], window=14)
        group['RSI'] = rsi.rsi()

        # 添加 MACD 和 Signal Line
        macd = MACD(close=group['收盘'], window_slow=26, window_fast=12, window_sign=9)
        group['MACD'] = macd.macd()
        group['MACD_SIGNAL'] = macd.macd_signal()

        df_list.append(group)
    return pd.concat(df_list)

# ===== 2. 预处理函数 =====
def preprocess(df, scaler=None, is_train=True):
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by=['股票代码', '日期'])
    df = add_technical_indicators(df)

    df['涨跌'] = ((df['收盘'] - df['开盘']) > 0).astype(int)
    df['成交量'] = np.log1p(df['成交量'])
    df['成交额'] = np.log1p(df['成交额'])

    df['原始股票代码'] = df['股票代码']

    features = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', 'RSI', 'MACD', 'MACD_SIGNAL']
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if is_train:
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        return df, features, scaler
    else:
        df[features] = scaler.transform(df[features])
        return df, features

# ===== 3. 创建序列样本 =====
def create_sequences(df, seq_len, features):
    X, y = [], []
    grouped = df.groupby('股票代码')
    for _, group in grouped:
        group = group.sort_values('日期')
        if len(group) <= seq_len:
            continue
        for i in range(len(group) - seq_len):
            seq = group.iloc[i:i + seq_len]
            label = group.iloc[i + seq_len]['涨跌']
            if not np.isnan(label):
                X.append(seq[features].values)
                y.append(label)
    return np.array(X), np.array(y)

# ===== 4. Transformer 模型 =====
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu"
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 自动处理 2D 输入，例如 [batch, features]
        if x.ndim == 2:
            x = x.unsqueeze(1)  # 变为 [batch, seq_len=1, features]
        x = self.input_proj(x)  # -> [batch, seq_len, d_model]
        x = self.transformer(x)  # -> [batch, seq_len, d_model]
        x = x[:, -1, :]  # 提取最后一个时间步
        x = self.fc(x)   # -> [batch, num_classes]
        return x

# ===== 5. 自定义 FocalLoss（二分类）=====
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

# ===== ⚠️ 自动检查 Transformer 输入张量维度和设备兼容性 =====
def check_model_io_shape(model, X_val, features):
    try:
        test_input = torch.tensor(X_val[:2], dtype=torch.float32).to(DEVICE)
        print("🟩 输入样本维度：", test_input.shape)

        # 检查模型结构是否能正常前向传播
        model.eval()
        with torch.no_grad():
            out = model(test_input)
        print("🟩 模型输出维度：", out.shape)

        # 输出维度检查
        if len(out.shape) != 1 or out.shape[0] != test_input.shape[0]:
            raise ValueError(f"❌ 模型输出维度异常: {out.shape}，请确认 forward 输出正确")

        print("✅ 模型结构与数据维度兼容，可以开始训练")
    except Exception as e:
        print("❌ 检查失败，请根据错误信息调整输入或模型结构：")
        raise e


# ===== 6. 模型训练 =====
def train_model(model, X_train, y_train, X_val, y_val):
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR,
                                                    steps_per_epoch=len(X_train)//BATCH_SIZE+1, epochs=EPOCHS)

    train_losses, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for i in range(0, len(X_train), BATCH_SIZE):
            xb = torch.tensor(X_train[i:i + BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_train[i:i + BATCH_SIZE], dtype=torch.float32).to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(xb)
            pred_label = (torch.sigmoid(pred) > 0.5).int()
            correct += (pred_label == yb.int()).sum().item()
            total += len(xb)

        acc = correct / total
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE))
            val_label = (torch.sigmoid(val_pred) > 0.5).int()
            val_acc = (val_label.cpu().numpy() == y_val).mean()

        train_losses.append(total_loss / len(X_train))
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")

    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()

# ===== 7. 预测并导出结果 =====
def predict_latest(df, model, seq_len, features):
    pred_list = []
    grouped = df.groupby('股票代码')

    for code, group in grouped:
        group = group.sort_values('日期')
        if len(group) >= seq_len:
            seq = group.iloc[-seq_len:][features].values
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(seq_tensor).cpu().item()

            raw_code = group.iloc[-1]['原始股票代码']
            pred_list.append((raw_code, pred))

    pred_sorted = sorted(pred_list, key=lambda x: x[1], reverse=True)
    return pred_sorted

# ===== 8. 主流程 =====
train = pd.read_csv('train.csv')
train, features, scaler = preprocess(train, is_train=True)
X, y = create_sequences(train, SEQ_LEN, features)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = TimeSeriesTransformer(input_dim=10, d_model=64, nhead=4, num_layers=2, num_classes=2).to(DEVICE)
print(f"使用设备: {DEVICE}")

# 确保输入 shape 正确
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
output = model(X_val_tensor)  # 自动处理维度

check_model_io_shape(model, X_val, features)
train_model(model, X_train, y_train, X_val, y_val)

test = pd.read_csv('test.csv')
test, _ = preprocess(test, scaler=scaler, is_train=False)
prediction_result = predict_latest(test, model, SEQ_LEN, features)

top10 = [x[0] for x in prediction_result[:10]]
bottom10 = [x[0] for x in prediction_result[-10:]]
result = pd.DataFrame({'涨幅最大股票代码': top10, '涨幅最小股票代码': bottom10})
result.to_csv('result.csv', index=False, encoding='utf-8')
print("✅ 预测完成，结果保存为 result.csv")
