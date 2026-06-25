# Stock Ranking — LSTM 股票价格预测与排名系统

## 项目简介 (Project Overview)

本项目使用 **LSTM（长短期记忆网络）** 对股票价格进行预测，并根据预期价格变化率对股票进行排名。系统包含完整的特征工程流水线、模型训练和推理模块。

**技术栈：** Python, PyTorch, LSTM

---

# Stock Ranking — LSTM for Stock Price Prediction

LSTM-based stock price prediction model for ranking stocks by expected price change rate.

## Method
- **Input**: 32-day sliding window of stock features (open, close, high, low, volume, turnover)
- **Model**: 4-layer LSTM (hidden_size=500, dropout=0.2) + Linear output
- **Loss**: MSELoss
- **Optimizer**: Adam (lr=1e-4) + StepLR scheduler (γ=0.7 every 5 epochs)
- **Output**: Predicted next-day closing price → price change rate → ranking

## Pipeline
```
Raw CSV → Feature Engineering → Sliding Windows → LSTM Training → Inference → Ranking
```

## Quick Start
```bash
python featurework.py     # Feature engineering
python train.py           # Train LSTM model (5 epochs per stock)
python test.py            # Predict and rank stocks
```

## Project Structure
```
├── code/
│   ├── featurework.py    # CSV feature engineering
│   ├── train.py          # LSTM training
│   └── test.py           # Inference & ranking
├── LSTM.py / LSTM2.py    # Alternative implementations
├── Transformer.py        # Transformer baseline
├── Grok1.py              # Grok-1 style experiment
└── README.md
```

## License
MIT
