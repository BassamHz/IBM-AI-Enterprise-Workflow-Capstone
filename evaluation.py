import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def evaluate_model_performance(test):
    """
    Evaluate model performance using RMSE and MAPE.
    """
    print("\nEvaluating Model Performance...")
    if 'prophet_forecast' not in test.columns or test['prophet_forecast'].isna().all():
        print("Error: 'prophet_forecast' column missing or contains all NaNs.")
        print(f"Prophet Forecast values:\n{test.get('prophet_forecast', 'Column not found')}")
        return
    try:
        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(test['revenue'], test['prophet_forecast']))
        # Compute MAPE
        mape = mean_absolute_percentage_error(test['revenue'], test['prophet_forecast'])

        # Display results
        print(f"Performance Metrics:\n  RMSE: {rmse:.2f}\n  MAPE: {mape:.2%}")

        # Return results for further use
        return {"RMSE": rmse, "MAPE": mape}
    except Exception as e:
        print(f"Error during performance evaluation: {e}")
        return {"RMSE": None, "MAPE": None}


import numpy as np
import matplotlib.pyplot as plt

def detect_outliers(train, test, show_plots=True):
    """
    Detect outliers in the training data based on residuals from predictions.

    Parameters:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset containing the 'prophet_forecast' column.
        show_plots (bool): Whether to display or save plots.

    Returns:
        pd.DataFrame: Outliers detected in the dataset.
    """
    print("\nDetecting Outliers...")
    try:
        # Ensure 'prophet_forecast' exists in the test DataFrame
        if 'prophet_forecast' not in test.columns:
            raise ValueError("'prophet_forecast' column missing in test DataFrame.")

        # Drop rows with missing values in 'revenue' or 'prophet_forecast'
        test = test.dropna(subset=['revenue', 'prophet_forecast'])

        # Calculate residuals between actual and predicted revenue
        residuals = test['revenue'] - test['prophet_forecast']

        # Determine a threshold for outlier detection (e.g., 1.5 standard deviations)
        threshold = 1.5 * np.std(residuals)

        # Identify outliers where residuals exceed the threshold
        outliers = test[np.abs(residuals) > threshold]

        # Display outlier information
        print(f"Detected {len(outliers)} outliers:")
        print(outliers[['date', 'revenue']])

        # Plot outliers
        if show_plots or True:  # Ensure plots are always saved
            plt.figure(figsize=(12, 6))
            plt.plot(train['date'], train['revenue'], label='Training Revenue', color='blue', alpha=0.6)
            plt.plot(test['date'], test['revenue'], label='Test Revenue', color='green', alpha=0.6)
            plt.scatter(outliers['date'], outliers['revenue'], color='red', label='Outliers', zorder=5)
            plt.title('Revenue with Detected Outliers')
            plt.xlabel('Date')
            plt.ylabel('Revenue')
            plt.legend()
            plt.grid()

            # Save the plot
            output_path = "./output/outliers.png"
            plt.savefig(output_path, bbox_inches="tight")
            print(f"Outlier plot saved to {output_path}.")
            plt.close()
        else:
            print("Plots are disabled.")

        return outliers

    except Exception as e:
        print(f"Error during outlier detection: {e}")
        return None




def compare_predictions(test, show_plots=True):
    """
    Visualize and compare actual vs predicted values for the test set.

    Parameters:
        test (pd.DataFrame): Test dataset containing actual and predicted values.
        show_plots (bool): Whether to display or save plots.

    Returns:
        None
    """
    print("\nComparing Predictions with Actual Values...")
    try:
        # Create a plot to compare predictions with actual values
        plt.figure(figsize=(12, 6))
        plt.plot(test['date'], test['revenue'], label='Actual Revenue', color='blue')
        plt.plot(test['date'], test['prophet_forecast'], label='Predicted Revenue', color='green', linestyle='--')
        plt.title('Actual vs Predicted Revenue')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid()
        if show_plots:
            plt.show()
    except Exception as e:
        print(f"Error during prediction comparison: {e}")
        

def log_model_performance(model_name, rmse, mape, output_dir):
    """
    Log the performance of a model into a CSV file.
    """
    log_file = os.path.join(output_dir, "model_comparison.csv")
    
    # Check if the file exists
    if not os.path.exists(log_file):
        # Create a new DataFrame with the model's performance
        df = pd.DataFrame(columns=["Model", "RMSE", "MAPE", "Comments"])
    else:
        # Read the existing log file
        df = pd.read_csv(log_file)

    # Add the new row to the DataFrame
    new_row = pd.DataFrame({"Model": [model_name], "RMSE": [rmse], "MAPE": [mape], "Comments": [""]})
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the log file
    df.to_csv(log_file, index=False)
    print(f"Model performance logged: {model_name} - RMSE: {rmse}, MAPE: {mape}")



def compare_all_models(test, sarima_forecast, lstm_forecast, output_dir, show_plots=True):
    """
    Compare actual vs predicted values for Prophet, SARIMA, and LSTM.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test['date'], test['revenue'], label='Actual Revenue', color='blue')
    plt.plot(test['date'], test['prophet_forecast'], label='Prophet Forecast', color='green', linestyle='--')
    plt.plot(test['date'], sarima_forecast, label='SARIMA Forecast', color='orange', linestyle=':')
    plt.plot(test['date'][-len(lstm_forecast):], lstm_forecast, label='LSTM Forecast', color='red')
    plt.title('Model Comparison: Actual vs Predicted Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(file_path) if not show_plots else plt.show()
    print(f"Comparison plot saved to {file_path}")
