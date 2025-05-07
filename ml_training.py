import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Initial Draft of training the model on single ETFs.

# Load and preprocess data
data = pd.read_csv("etfs/AAAU.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Compute technical indicators
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['signal_line'] = data['MACD'].ewm(span=9, adjust=False).mean()

def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data)

# Define features and target
features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'MACD', 'signal_line', 'RSI']
data_features = data[features_list].dropna()
data['trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data_features['trend'] = data['trend'].dropna()

# Create sequences
seq_length = 15
X = []
y = []
for i in range(len(data_features) - seq_length):
    X.append(data_features.iloc[i:i+seq_length][features_list].values)
    y.append(data_features['trend'].iloc[i+seq_length-1])

X = np.array(X)
y = np.array(y)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, len(features_list))).reshape(X.shape)

# Split data
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(features_list))))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict next day
last_sequence = data_features.iloc[-seq_length:][features_list].values
last_sequence_scaled = scaler.transform(last_sequence.reshape(1, -1)).reshape(1, seq_length, len(features_list))
next_trend_pred = (model.predict(last_sequence_scaled) > 0.5).astype(int)[0][0]
print(f"Predicted Trend for Next Day: {'Green' if next_trend_pred == 1 else 'Red'}")