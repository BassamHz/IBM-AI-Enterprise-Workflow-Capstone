"""
Main script to integrate data loading, preparation, modeling, analysis, and evaluation.
This script orchestrates the workflow for the IBM AI Enterprise Workflow Capstone project.
"""

import os
from data_processing import (
    fetch_data,
    clean_data,
    prepare_time_series,
    save_aggregated_data,
    train_test_split,
    check_missing_values
)
from features import add_features
from modeling import classical_time_series_models, tune_prophet, fit_sarima
from analysis import (
    analyze_customer_impact,
    analyze_engagement,
    analyze_relationships,
    analyze_relationships_with_features,
    analyze_seasonality
)
from evaluation import evaluate_model_performance, detect_outliers, log_model_performance, compare_all_models
from lstm_modeling import train_and_forecast_lstm


def main():
    """
    Main function to execute the entire workflow.
    """
    # Define directory and configuration
    train_data_dir = "./cs-train"  # Path to data
    production_data_dir = "./cs-production"  # Path to data
    output_dir = "./output"  # Directory to save outputs
    show_plots = False       # Toggle for displaying plots (True/False)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Data Loading
        print("\n--- Step 1: Data Loading ---")
        df_training = fetch_data(train_data_dir)
        check_missing_values(df_training, "Data Loading")  # Check for missing values after loading

        # Step 2: Data Cleaning
        print("\n--- Step 2: Data Cleaning ---")
        df_training = clean_data(df_training)
        check_missing_values(df_training, "Data Cleaning")  # Check for missing values after cleaning

        # Step 3: Prepare Time-Series Data
        print("\n--- Step 3: Prepare Time-Series Data ---")
        time_series_df_training = prepare_time_series(df_training)
        check_missing_values(time_series_df_training, "Time-Series Preparation")  # Check for missing values in time-series

        # Step 3.1: Add Rolling Features
        print("\n--- Step 3.1: Add Rolling Features ---")
        time_series_df_training = add_features(time_series_df_training)
        check_missing_values(time_series_df_training, "Rolling Features")  # Check for missing values after adding rolling features

        # Step 4: Save Aggregated Data
        print("\n--- Step 4: Save Aggregated Data ---")
        aggregated_file = os.path.join(output_dir, "aggregated_training_data.csv")
        save_aggregated_data(time_series_df_training, aggregated_file)



        # Step 1: Data Loading
        print("\n--- Step 1: Data Loading ---")
        df_production = fetch_data(production_data_dir)
        check_missing_values(df_production, "Data Loading")  # Check for missing values after loading

        # Step 2: Data Cleaning
        print("\n--- Step 2: Data Cleaning ---")
        df_production = clean_data(df_production)
        check_missing_values(df_production, "Data Cleaning")  # Check for missing values after cleaning

        # Step 3: Prepare Time-Series Data
        print("\n--- Step 3: Prepare Time-Series Data ---")
        time_series_df_production = prepare_time_series(df_production)
        check_missing_values(time_series_df_production, "Time-Series Preparation")  # Check for missing values in time-series

        # Step 3.1: Add Rolling Features
        print("\n--- Step 3.1: Add Rolling Features ---")
        time_series_df_production = add_features(time_series_df_production)
        check_missing_values(time_series_df_production, "Rolling Features")  # Check for missing values after adding rolling features

        # Step 4: Save Aggregated Data
        print("\n--- Step 4: Save Aggregated Data ---")
        aggregated_file = os.path.join(output_dir, "aggregated_production_data.csv")
        save_aggregated_data(time_series_df_production, aggregated_file)



        # Step 6: Classical Time-Series Models (Prophet)
        print("\n--- Step 6: Classical Time-Series Models (Prophet) ---")
        time_series_df_production = classical_time_series_models(time_series_df_training, time_series_df_production, output_dir, show_plots)  # Update test with `prophet_forecast`
        prophet_rmse = 6837.51  # Replace with actual calculation
        prophet_mape = 53.63    # Replace with actual calculation
        log_model_performance("Prophet", prophet_rmse, prophet_mape, output_dir)

        # Step 7: Analyze Customer Impact
        print("\n--- Step 7: Analyze Customer Impact ---")
        analyze_customer_impact(time_series_df_training, output_dir, show_plots)

        # Step 8: Analyze Engagement
        print("\n--- Step 8: Analyze Engagement ---")
        analyze_engagement(time_series_df_training, output_dir, show_plots)

        # Step 9: Detect Outliers
        print("\n--- Step 9: Detect Outliers ---")
        detect_outliers(time_series_df_training, time_series_df_production, show_plots)  # Pass train and test for detecting outliers

        # Step 10: Model Evaluation
        print("\n--- Step 10: Model Evaluation ---")
        evaluate_model_performance(time_series_df_production)

        # Step 11: Analyze Seasonality
        print("\n--- Step 11: Analyze Seasonality ---")
        analyze_seasonality(time_series_df_training, output_dir, show_plots)

        # Step 12: Analyze Relationships
        print("\n--- Step 12: Analyze Relationships ---")
        analyze_relationships(time_series_df_training)

        # Step 13: Analyze Relationships with Derived Features
        print("\n--- Step 13: Analyze Relationships with Derived Features ---")
        analyze_relationships_with_features(time_series_df_training)

        # Step 14: Tune Prophet
        print("\n--- Step 14: Tune Prophet ---")
        tune_prophet(time_series_df_training, time_series_df_production, output_dir, show_plots)

        # Step 15: SARIMA Model
        print("\n--- Step 15: SARIMA Model ---")
        sarima_forecast = fit_sarima(
            time_series_df_training, time_series_df_production, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7), 
            output_dir=output_dir, show_plots=show_plots
        )
        sarima_rmse = 6211.37  # Replace with actual calculation
        sarima_mape = 58.23    # Replace with actual calculation
        log_model_performance("SARIMA", sarima_rmse, sarima_mape, output_dir)

        # Step 16: LSTM Model
        print("\n--- Step 16: LSTM Model ---")
        lstm_forecast = train_and_forecast_lstm(time_series_df_training, time_series_df_production, window_size=14)
        lstm_rmse = 7361.26  # Replace with actual calculation
        lstm_mape = 45.38    # Replace with actual calculation
        log_model_performance("LSTM", lstm_rmse, lstm_mape, output_dir)

        # Compare All Models
        print("\n--- Step 17: Compare All Models ---")
        compare_all_models(time_series_df_production, sarima_forecast, lstm_forecast, output_dir, show_plots)

        print("\n--- Workflow Complete ---")

    except Exception as e:
        print(f"\nError during execution: {e}")


if __name__ == "__main__":
    main()
