import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Data Preprocessing
def preprocess_data(symbol, start_date, end_date, sequence_length):
    # Download stock data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate additional features
    data['Returns'] = data['Close'].pct_change()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Drop NaN values
    data = data.dropna()
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Returns', 'MA5', 'MA20', 'RSI']])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), :])
        y.append(scaled_data[i + sequence_length, 0])
    
    return np.array(X), np.array(y), scaler

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Transformer Model
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(StockTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.FloatTensor(X_train[i:i+batch_size])
            batch_y = torch.FloatTensor(y_train[i:i+batch_size])
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs[:, -1, 0], batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs[:, -1, 0], torch.FloatTensor(y_val))
        
        train_losses.append(train_loss / len(X_train))
        val_losses.append(val_loss.item())
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

# Main execution
if __name__ == "__main__":
    # Set parameters
    symbol = "^GSPC"  # S&P 500
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    sequence_length = 60
    
    # Preprocess data
    X, y, scaler = preprocess_data(symbol, start_date, end_date, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize model
    model = StockTransformer(input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2, dim_feedforward=256)
    
    # Train model
    epochs = 50
    batch_size = 32
    train_losses, val_losses = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_test))[:, -1, 0].numpy()
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros_like(predictions))))[:, 0]
    actual = scaler.inverse_transform(np.column_stack((y_test, np.zeros_like(y_test))))[:, 0]
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(actual, predictions)
    print(f"MAPE: {mape:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('S&P 500 Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
