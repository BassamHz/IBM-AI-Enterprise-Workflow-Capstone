"""
Modeling Module
Contains functions to build, train, and tune time-series models.
"""

import os
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools



def classical_time_series_models(train, test, output_dir, show_plots=True):
    """
    Apply Prophet to the training data and optionally display plots.
    """
    print("\nProphet Model:")
    try:
        # Rename columns for Prophet compatibility
        prophet_train = train.rename(columns={'date': 'ds', 'revenue': 'y'})

        # Initialize and fit Prophet model
        prophet_model = Prophet()
        prophet_model.fit(prophet_train)

        # Create a future DataFrame for predictions
        future = prophet_model.make_future_dataframe(periods=len(test))
        forecast = prophet_model.predict(future)

        # Debugging forecast and test alignment
        print("\n--- Debugging Forecast DataFrame ---")
        print(f"Forecast shape: {forecast.shape}")
        print(f"Forecast tail:\n{forecast.tail()}")
        print(f"Test shape: {test.shape}")
        print(f"Test head:\n{test.head()}")

        # Ensure correct alignment between forecast and test
        if len(forecast.tail(len(test))) == len(test):
            print("Forecast and Test DataFrame lengths align correctly.")

            # Safely assign the `prophet_forecast` column
            test = test.copy()  # Avoid SettingWithCopyWarning
            test['prophet_forecast'] = forecast['yhat'][-len(test):].values

            # Debugging: Confirm column addition
            print("\nProphet Forecast values successfully added to Test DataFrame.")
            print(f"Test DataFrame columns: {test.columns}")
        else:
            raise ValueError("Mismatch between forecast and test data lengths.")

        # Optional: Plot the forecast
        plt.figure(figsize=(10, 5))
        plt.plot(train['date'], train['revenue'], label='Training Data')
        plt.plot(test['date'], test['revenue'], label='Actual Test Data')
        plt.plot(test['date'], test['prophet_forecast'], label='Prophet Forecast', color='green')
        plt.legend()
        plt.title('Prophet Model Forecast')
        file_path = os.path.join(output_dir, "prophet_forecast.png")
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
        plt.close()

    except Exception as e:
        print(f"Prophet Error: {e}")
        # Explicitly add a column with NaNs in case of failure
        test['prophet_forecast'] = np.nan
        print("Failed to add Prophet Forecast. Column initialized with NaNs.")

    return test  # Return the updated test DataFrame


def tune_prophet(train, test, output_dir, show_plots=True):
    """
    Tune Prophet hyperparameters and evaluate performance.
    """
    try:
        # Prepare training data for Prophet
        prophet_train = train.rename(columns={'date': 'ds', 'revenue': 'y'})
        
        # Handle missing values in regressors
        prophet_train['unique_customers'] = prophet_train['unique_customers'].fillna(0)
        prophet_train['total_views'] = prophet_train['total_views'].fillna(0)

        # Initialize and tune the model
        tuned_model = Prophet(
            changepoint_prior_scale=0.5,  # Adjust flexibility for changepoints
            seasonality_prior_scale=10,   # Adjust seasonal flexibility
            yearly_seasonality=True
        )
        
        # Add custom regressors
        tuned_model.add_regressor('unique_customers')
        tuned_model.add_regressor('total_views')

        # Fit the tuned model
        tuned_model.fit(prophet_train)

        # Prepare test data for prediction
        future = tuned_model.make_future_dataframe(periods=len(test))
        future = future.merge(
            test[['date', 'unique_customers', 'total_views']].rename(columns={'date': 'ds'}),
            on='ds',
            how='left'
        )

        # Handle NaNs in future DataFrame
        future['unique_customers'] = future['unique_customers'].fillna(0)
        future['total_views'] = future['total_views'].fillna(0)

        # Predict with tuned model
        forecast = tuned_model.predict(future)

        # Evaluate performance
        rmse = np.sqrt(mean_squared_error(test['revenue'], forecast['yhat'][-len(test):]))
        mape = mean_absolute_percentage_error(test['revenue'], forecast['yhat'][-len(test):])

        print(f"\nTuned Prophet Model Metrics:\nRMSE: {rmse:.2f}\nMAPE: {mape:.2%}")

        # Save or show the plot
        file_path = os.path.join(output_dir, "tuned_prophet_forecast.png")
        tuned_model.plot(forecast)
        plt.savefig(file_path) if not show_plots else plt.show()
        print(f"Plot saved to {file_path}")
        plt.close()

    except Exception as e:
        print(f"Error during Prophet tuning: {e}")


def fit_sarima(train, test, order, seasonal_order, output_dir, show_plots=False):
    """
    Fit and forecast using SARIMA, then save the plot instead of displaying it.
    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
        order (tuple): SARIMA order (p, d, q).
        seasonal_order (tuple): SARIMA seasonal order (P, D, Q, S).
        output_dir (str): Directory to save the plot.
        show_plots (bool): Whether to display the plot.
    Returns:
        np.array: Forecasted values.
    """
    try:
        # Fit SARIMA
        model = SARIMAX(train['revenue'], order=order, seasonal_order=seasonal_order)
        sarima_model = model.fit(disp=False, maxiter=500)

        # Forecast
        forecast = sarima_model.forecast(steps=len(test))

        # Debugging
        print(f"SARIMA Forecast: {forecast[:5]}")  # Print first 5 forecasted values

        # Evaluate
        rmse = np.sqrt(mean_squared_error(test['revenue'], forecast))
        mape = mean_absolute_percentage_error(test['revenue'], forecast)
        print(f"SARIMA Performance - RMSE: {rmse:.2f}, MAPE: {mape:.2%}")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(train['date'], train['revenue'], label='Training Data')
        plt.plot(test['date'], test['revenue'], label='Test Data')
        plt.plot(test['date'], forecast, label='SARIMA Forecast', color='green')
        plt.legend()
        plt.title('SARIMA Model Forecast')

        # Save the plot
        file_path = os.path.join(output_dir, "sarima_forecast.png")
        plt.savefig(file_path) if not show_plots else plt.show()
        print(f"Plot saved to {file_path}")
        plt.close()

        return sarima_model

    except Exception as e:
        print(f"Error in SARIMA model: {e}")
        return np.nan


def optimize_sarima(train, p_values, d_values, q_values, P_values, D_values, Q_values, s):
    best_score, best_cfg = float("inf"), None
    for param in itertools.product(p_values, d_values, q_values):
        for seasonal_param in itertools.product(P_values, D_values, Q_values):
            try:
                model = SARIMAX(train['revenue'], order=param, seasonal_order=seasonal_param + (s,))
                results = model.fit(disp=False)
                aic = results.aic
                if aic < best_score:
                    best_score, best_cfg = aic, (param, seasonal_param)
            except:
                continue
    print(f"Best SARIMA{best_cfg[0]}x{best_cfg[1]} - AIC:{best_score}")
    return best_cfg
