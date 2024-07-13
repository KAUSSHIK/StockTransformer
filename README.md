# Stock Price Prediction Using Transformers

## Overview

This project implements a stock price prediction model using a custom Transformer architecture. It focuses on predicting S&P 500 stock prices using historical data, technical indicators, and other financial metrics. The model is built using PyTorch and incorporates time series analysis techniques.

## Features

- Custom Transformer architecture for time series prediction
- Data preprocessing pipeline for handling financial time series data
- Integration of multiple data sources (price, returns, moving averages, RSI)
- Training and evaluation framework
- Visualization of predictions and model performance

## Requirements

- Python 3.7+
- PyTorch
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/KAUSSHIK/StockTransformer.git
   cd stock-price-prediction
   ```

2. Install the required packages.

## Usage

To run the stock price prediction model:

```
python model.py
```

This script will:
1. Download S&P 500 historical data
2. Preprocess the data and calculate technical indicators
3. Train the Transformer model
4. Evaluate the model's performance
5. Generate plots of the predictions and training/validation loss

## Model Architecture

The model uses a Transformer architecture adapted for time series prediction:
- Embedding layer to project input features into a higher-dimensional space
- Positional encoding to capture sequential information
- Transformer encoder layers for processing the time series data
- Linear decoder to produce the final prediction

## Data Preprocessing

The preprocessing pipeline includes:
- Downloading stock data using yfinance
- Calculating additional features (returns, moving averages, RSI)
- Normalizing the data using MinMaxScaler
- Creating sequences for the model input

## Results

The model's performance is evaluated using Mean Absolute Percentage Error (MAPE). The script generates two plots:
1. Actual vs. Predicted stock prices
2. Training and Validation loss over epochs

## Future Improvements

- Implement ARIMA and LSTM models for baseline comparison
- Incorporate more financial indicators and external data sources
- Optimize hyperparameters using techniques like grid search or Bayesian optimization
- Implement real-time prediction capabilities

## Contributing

Contributions to improve the model or extend its capabilities are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is open-source and available under the MIT License.
