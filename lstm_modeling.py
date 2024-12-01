import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def prepare_lstm_data(df, features, target, window_size):
    """
    Prepare time-series data for LSTM using multiple features.
    Parameters:
        df (pd.DataFrame): Input DataFrame with features.
        features (list): List of feature column names to use as input.
        target (str): Target column name.
        window_size (int): Number of timesteps in the input sequence.
    """
    X, y = [], []
    feature_data = df[features].values
    target_data = df[target].values
    for i in range(len(df) - window_size):
        X.append(feature_data[i:i+window_size])
        y.append(target_data[i+window_size])
    return np.array(X), np.array(y)


def train_and_forecast_lstm(train, test, window_size):
    """
    Train an LSTM model and forecast values for the test dataset.

    Args:
        train (DataFrame): Training dataset.
        test (DataFrame): Testing dataset.
        window_size (int): Size of the input window for the LSTM model.

    Returns:
        np.array: Forecasted values.
    """
    # Prepare the data for LSTM
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    # Extract revenue as the feature to predict
    train_revenue = train["revenue"].values
    test_revenue = test["revenue"].values

    # Normalize the data (min-max scaling)
    min_val = min(train_revenue)
    max_val = max(train_revenue)
    train_revenue = (train_revenue - min_val) / (max_val - min_val)
    test_revenue = (test_revenue - min_val) / (max_val - min_val)

    # Create sequences
    X_train, y_train = create_sequences(train_revenue, window_size)
    X_test, y_test = create_sequences(np.concatenate([train_revenue[-window_size:], test_revenue]), window_size)

    # Reshape for LSTM (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Forecast
    predictions = model.predict(X_test).flatten()

    # Reverse normalization
    predictions = predictions * (max_val - min_val) + min_val
    y_test = y_test * (max_val - min_val) + min_val

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Test Data")
    plt.plot(predictions, label="LSTM Forecast", color="red")
    plt.legend()
    plt.title("LSTM Model Forecast")
    plt.savefig("./output/lstm_forecast.png")
    plt.close()

    print(f"LSTM Performance - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return predictions
