import pandas as pd

"""
Feature Engineering Module
Contains functions to engineer features for the dataset.
"""

def add_rolling_features(df):
    """
    Add rolling average and standard deviation features to the dataset.

    Parameters:
        df (pd.DataFrame): Dataframe containing the time-series data.

    Returns:
        pd.DataFrame: Updated dataframe with rolling average and standard deviation features.
    """
    print("\nAdding Rolling Features...")
    try:
        # Calculate a 7-day rolling mean for revenue
        df['rolling_mean'] = df['revenue'].rolling(window=7).mean()
        # Calculate a 7-day rolling standard deviation for revenue
        df['rolling_std'] = df['revenue'].rolling(window=7).std()

        # Fill missing values with column mean
        df[['rolling_mean', 'rolling_std']] = df[['rolling_mean', 'rolling_std']].fillna(method='bfill')
        print("Rolling features added successfully!")

    except Exception as e:
        print(f"Error adding rolling features: {e}")
    return df


def add_lag_features(df, fill_value = None):
    """
    Add lag features (previous values) to capture temporal dependencies.

    Parameters:
        df (pd.DataFrame): Dataframe containing the time-series data.

    Returns:
        pd.DataFrame: Updated dataframe with lag features.
    """
    try:
        # Add revenue from the previous day
        df['revenue_lag_1'] = df['revenue'].shift(1)
        # Add revenue from one week prior
        df['revenue_lag_7'] = df['revenue'].shift(7)

        if fill_value is not None:
            df[['revenue_lag_1', 'revenue_lag_7']].fillna(fill_value, inplace=True)
        print("Lag features added successfully!")

    except Exception as e:
        print(f"Error adding lag features: {e}")
    return df


def add_derived_features(df):
    """
    Add derived features such as revenue per customer and views per customer.

    Parameters:
        df (pd.DataFrame): Dataframe containing the time-series data.

    Returns:
        pd.DataFrame: Updated dataframe with derived features.
    """
    try:
        # Calculate revenue per customer
        df['revenue_per_customer'] = df['revenue'] / (df['unique_customers'] + 1e-6)  # Avoid division by zero
        # Calculate total views per customer
        df['views_per_customer'] = df['total_views'] / (df['unique_customers'] + 1e-6)
        print("Derived features added successfully!")
    except Exception as e:
        print(f"Error adding derived features: {e}")
    return df


def add_features(df):
    """
    Add all rolling, lag, and derived features to the dataset.

    Parameters:
        df (pd.DataFrame): Dataframe containing the time-series data.

    Returns:
        pd.DataFrame: Updated dataframe with all features.
    """
    print("\nAdding All Features...")
    try:
        df = add_rolling_features(df)
        df = add_lag_features(df)
        df = add_derived_features(df)
        print("All features added successfully!")
    except Exception as e:
        print(f"Error adding features: {e}")
    return df
