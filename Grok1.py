import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Custom Dataset for stock data
class StockDataset(Dataset):
    def __init__(self, data, seq_len=20, target_col='涨跌幅'):
        self.data = data
        self.seq_len = seq_len
        self.target_col = target_col
        self.features = ['开盘', '收盘', '成交量', '简单移动平均', '相对强弱指数']

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[self.features].iloc[idx:idx + self.seq_len].values
        y = self.data[self.target_col].iloc[idx + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, lstm_output):
        batch_size = lstm_output.size(0)
        seq_len = lstm_output.size(1)
        hidden = lstm_output
        energy = torch.tanh(self.attention(hidden))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention_weights = torch.bmm(v, energy).squeeze(1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        return context, attention_weights


# CNN-LSTM-Attention Model
class StockPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super(StockPredictionModel, self).__init__()
        self.hidden_dim = hidden_dim

        # CNN Layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)

        # LSTM Layer
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # CNN
        x = x.transpose(1, 2)  # Adjust for Conv1d [batch, channels, seq]
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # Back to [batch, seq, features] for LSTM

        # LSTM
        x, _ = self.lstm1(x)  # Returns sequence for all time steps
        x, _ = self.lstm2(x)  # Returns only the last time step by default

        # Attention
        context, attention_weights = self.attention(x)
        context = context.squeeze(1)

        # Output
        out = self.fc(context)
        return out


# Training Function
def train_model(model, train_loader, val_loader, epochs=40, batch_size=64, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')


# Prediction Function
def predict_top_stocks(model, data, top_n=10):
    model.eval()
    predictions = []
    stock_codes = data['股票代码'].unique()

    with torch.no_grad():
        for code in stock_codes:
            code_data = data[data['股票代码'] == code].copy()
            if len(code_data) <= 20:
                continue
            dataset = StockDataset(code_data)
            loader = DataLoader(dataset, batch_size=1)
            code_preds = []
            for x, _ in loader:
                x = x.to(next(model.parameters()).device)
                pred = model(x)
                code_preds.append(pred.item())
            avg_pred = np.mean(code_preds[-top_n:])  # Average of last 10 predictions
            predictions.append((code, avg_pred))

    # Sort by predicted change and get top 10 max and min
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_max = [code for code, _ in predictions[:top_n]]
    top_min = [code for code, _ in predictions[-top_n:]]

    return top_max, top_min


# Main Execution
if __name__ == "__main__":
    # Load and preprocess data from train.csv
    train_data = pd.read_csv('train.csv')

    # Calculate technical indicators (SMA and RSI) for train data
    train_data['简单移动平均'] = train_data['收盘'].rolling(window=10).mean()
    delta = train_data['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    train_data['相对强弱指数'] = 100 - (100 / (1 + rs))
    train_data = train_data.dropna()

    # Split train data into train and validation sets
    train_size = int(0.8 * (len(train_data) - 20))
    train_set = train_data[:train_size + 20]
    val_set = train_data[train_size + 20:]

    # Create datasets and dataloaders
    train_dataset = StockDataset(train_set)
    val_dataset = StockDataset(val_set)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Initialize model
    model = StockPredictionModel(input_dim=5)  # 5 features: 开盘, 收盘, 成交量, SMA, RSI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train model
    train_model(model, train_loader, val_loader)

    # Load and preprocess test.csv
    test_data = pd.read_csv('test.csv')

    # Calculate technical indicators (SMA and RSI) for test data
    test_data['简单移动平均'] = test_data['收盘'].rolling(window=10).mean()
    delta = test_data['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    test_data['相对强弱指数'] = 100 - (100 / (1 + rs))
    test_data = test_data.dropna()

    # Predict top 10 stocks using test data
    top_max_stocks, top_min_stocks = predict_top_stocks(model, test_data)

    # Output results in the required format
    result = pd.DataFrame({
        '涨幅最大股票代码': top_max_stocks,
        '涨幅最小股票代码': top_min_stocks
    })
    result.to_csv('result2.csv', index=False)
    print(result)