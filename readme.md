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
