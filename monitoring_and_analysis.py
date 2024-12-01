import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_processing import fetch_data, clean_data, prepare_time_series

# API endpoints
TRAIN_URL = "http://127.0.0.1:5000/train"
PREDICT_URL = "http://127.0.0.1:5000/predict"
LOGFILE_URL = "http://127.0.0.1:5000/logfile"

def get_actual_values(data_dir):
    """
    Load and prepare actual revenue values from production JSON files.

    Args:
        data_dir (str): Directory containing the production JSON files.

    Returns:
        dict: A dictionary with dates as keys and actual revenue as values.
    """
    raw_data = fetch_data(data_dir)
    clean_df = clean_data(raw_data)
    time_series = prepare_time_series(clean_df)

    # Ensure proper date formatting
    time_series['date'] = pd.to_datetime(time_series['date']).dt.strftime('%Y-%m-%d')

    # Convert to dictionary
    actual_values = time_series.set_index('date')['revenue'].to_dict()
    return actual_values

def simulate_monitoring(train_dir, production_dir):
    """
    Simulate daily predictions, weekly re-training, and logging.

    Args:
        train_dir (str): Directory containing training JSON files.
        production_dir (str): Directory containing production JSON files.
    """
    # Train the model
    print(f"Training model using data from: {train_dir}")
    train_response = requests.post(TRAIN_URL)
    print("Training response:", train_response.json())

    # Fetch actual values
    actual_values = get_actual_values(production_dir)

    # Get date range from production data
    date_range = sorted(actual_values.keys())
    start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
    end_date = datetime.strptime(date_range[-1], '%Y-%m-%d')

    current_date = start_date
    all_predictions = []
    weekly_actuals = []
    weekly_predictions = []

    while current_date <= end_date:
        # Prepare prediction payload
        predict_payload = {
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": current_date.strftime("%Y-%m-%d")
        }
        response = requests.post(PREDICT_URL, json=predict_payload)
        prediction = response.json().get("predictions", [None])[0]

        # Fetch actual value
        actual = actual_values.get(current_date.strftime("%Y-%m-%d"))
        all_predictions.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "prediction": prediction,
            "actual": actual
        })

        # Append for weekly evaluation
        weekly_predictions.append(prediction)
        weekly_actuals.append(actual)

        # Perform weekly retraining and logging
        if current_date.weekday() == 6:  # Sunday
            if all(v is not None for v in weekly_actuals):
                weekly_rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(weekly_predictions, weekly_actuals)]))
                weekly_mape = np.mean([abs((a - p) / a) for p, a in zip(weekly_predictions, weekly_actuals)]) * 100
                print(f"Weekly RMSE: {weekly_rmse:.2f}, MAPE: {weekly_mape:.2f}%")

                log_payload = {
                    "Model": "Simulated Monitor",
                    "RMSE": weekly_rmse,
                    "MAPE": weekly_mape,
                    "Comments": f"Week ending {current_date.strftime('%Y-%m-%d')}"
                }
                requests.post(LOGFILE_URL, json=log_payload)

            print(f"Retraining model using data from {train_dir} on {current_date.strftime('%Y-%m-%d')}")
            train_response = requests.post(TRAIN_URL)
            print("Retrain response:", train_response.json())

            weekly_predictions.clear()
            weekly_actuals.clear()

        current_date += timedelta(days=1)

    # Save predictions
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("simulated_predictions.csv", index=False)
    print("Simulated predictions saved to simulated_predictions.csv")

def post_production_analysis(predictions_file):
    """
    Compare predicted and actual values, evaluate model drift.

    Args:
        predictions_file (str): Path to the CSV file containing predictions and actuals.
    """
    df = pd.read_csv(predictions_file)

    # Drop missing actuals
    df.dropna(subset=["actual"], inplace=True)

    # Calculate performance metrics
    rmse = np.sqrt(np.mean((df["prediction"] - df["actual"]) ** 2))
    mape = np.mean(abs((df["actual"] - df["prediction"]) / df["actual"])) * 100
    print(f"Overall RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Plot predictions vs actuals
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["actual"], label="Actual", marker="o")
    plt.plot(df["date"], df["prediction"], label="Prediction", marker="x")
    plt.title("Predicted vs Actual Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("post_production_analysis.png")
    print("Post-production analysis plot saved to post_production_analysis.png")

if __name__ == "__main__":
    simulate_monitoring("./cs-train", "./cs-production")
    post_production_analysis("simulated_predictions.csv")
