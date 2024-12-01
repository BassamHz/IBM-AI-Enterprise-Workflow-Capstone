import os
import matplotlib
matplotlib.use('Agg')
from joblib import dump, load
import pandas as pd
from flask import Flask, request, jsonify
from data_processing import (
    fetch_data,
    clean_data,
    prepare_time_series,
    save_aggregated_data,
    check_missing_values
)
from features import add_features
from modeling import classical_time_series_models, fit_sarima, tune_prophet
from evaluation import evaluate_model_performance, detect_outliers, log_model_performance
from lstm_modeling import train_and_forecast_lstm
from analysis import (
    analyze_customer_impact,
    analyze_engagement,
    analyze_seasonality,
    analyze_relationships,
    analyze_relationships_with_features
)

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Step 1: Load Data
        training_data_dir = "./cs-train"
        df_training = fetch_data(training_data_dir)
        check_missing_values(df_training, "Data Loading")

        # Step 2: Clean Data
        ddf_trainingf = clean_data(df_training)
        check_missing_values(df_training, "Data Cleaning")

        # Step 3: Prepare Time-Series Data
        time_series_df_training = prepare_time_series(df_training)
        check_missing_values(time_series_df_training, "Time-Series Preparation")

        # Step 3.1: Add Rolling Features
        time_series_df_training = add_features(time_series_df_training)
        check_missing_values(time_series_df_training, "Rolling & Derived Features")

        # Step 4: Save Aggregated Data
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        aggregated_file = os.path.join(output_dir, "training_aggregated_data.csv")
        save_aggregated_data(time_series_df_training, aggregated_file)



        # Step 1: Load Data
        production_data_dir = "./cs-production"
        df_production = fetch_data(production_data_dir)
        check_missing_values(df_production, "Data Loading")

        # Step 2: Clean Data
        df_production = clean_data(df_production)
        check_missing_values(df_production, "Data Cleaning")

        # Step 3: Prepare Time-Series Data
        time_series_df_production = prepare_time_series(df_production)
        check_missing_values(time_series_df_production, "Time-Series Preparation")

        # Step 3.1: Add Rolling Features
        time_series_df_production = add_features(time_series_df_production)
        check_missing_values(time_series_df_production, "Rolling & Derived Features")

        # Step 4: Save Aggregated Data
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        aggregated_file = os.path.join(output_dir, "production_aggregated_data.csv")
        save_aggregated_data(time_series_df_production, aggregated_file)


        # Step 6: Classical Time-Series Models (Prophet)
        time_series_df_production = classical_time_series_models(time_series_df_training, time_series_df_production, output_dir, show_plots=False)

        # Step 7: SARIMA Model
        sarima_model = fit_sarima(
            time_series_df_training, time_series_df_production, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
            output_dir=output_dir, show_plots=False
        )
        sarima_forecast = sarima_model.forecast(steps=len(time_series_df_training))  # Optional: Use for evaluation

        # Step 8: LSTM Model
        lstm_forecast = train_and_forecast_lstm(time_series_df_training, time_series_df_production, window_size=14)

        # Step 9: Evaluate Models
        evaluate_model_performance(time_series_df_production)

        # Step 10: Analyze Engagement & Impact
        analyze_customer_impact(time_series_df_training, output_dir, show_plots=False)
        analyze_engagement(time_series_df_training, output_dir, show_plots=False)

        # Step 11: Detect Outliers
        detect_outliers(time_series_df_training, time_series_df_production, show_plots=False)

        # Step 12: Analyze Seasonality
        analyze_seasonality(time_series_df_training, output_dir, show_plots=False)

        # Step 13: Analyze Relationships
        analyze_relationships(time_series_df_training)

        # Step 14: Analyze Relationships with Derived Features
        analyze_relationships_with_features(time_series_df_training)

        # Step 15: Tune Prophet
        tune_prophet(time_series_df_training, time_series_df_production, output_dir, show_plots=False)

        # Log model performance (replace with actual RMSE and MAPE)
        log_model_performance("SARIMA", 6211.37, 58.23, output_dir)
        log_model_performance("LSTM", 7361.26, 45.38, output_dir)

        # Save models after training
        # Save the fitted SARIMA model
        dump(sarima_model, os.path.join(output_dir, "sarima_model.pkl"))
        dump(lstm_forecast, os.path.join(output_dir, "lstm_model.pkl"))


        return jsonify({"message": "Model training completed successfully!"}), 200

    except Exception as e:
        return jsonify({"error": f"Error during execution: {e}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        start_date = data['start_date']
        end_date = data['end_date']

        # Load SARIMA model
        sarima_model = load("./output/sarima_model.pkl")

        # Generate predictions
        prediction_dates = pd.date_range(start=start_date, end=end_date)
        forecast = sarima_model.forecast(steps=len(prediction_dates))

        # Convert predictions to JSON-serializable format
        forecast = forecast.tolist()  # Convert ndarray to list

        return jsonify({"predictions": forecast}), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/logfile', methods=['GET'])
def get_logfile():
    """
    Endpoint to return the model comparison log as JSON.
    """
    try:
        # Path to the CSV log file
        logfile_path = "./output/model_comparison.csv"
        
        # Check if the file exists
        if not os.path.exists(logfile_path):
            return jsonify({"error": "Log file not found. Please train the model first."}), 404

        # Read the CSV log file
        log_df = pd.read_csv(logfile_path)

        # Convert DataFrame to a JSON-friendly format
        log_json = log_df.to_dict(orient="records")

        return jsonify({"logfile": log_json}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve log file: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

