"""
Data Processing Module
Contains functions to load, clean, and prepare the dataset.
"""

import os
import pandas as pd


def fetch_data(data_dir):
    """
    Load JSON files from the directory and combine them into a single DataFrame.

    Args:
        data_dir (str): Directory containing JSON files.

    Returns:
        pd.DataFrame: Combined DataFrame of all files.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist!")

    # Find all JSON files in the directory
    json_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.json')]
    
    if not json_files:
        raise ValueError("No JSON files found in the directory!")

    # Load and combine all JSON files
    data_frames = []
    for file in json_files:
        df = pd.read_json(file)
        # Standardize column names
        df.columns = df.columns.str.lower()
        # Map non-uniform feature names
        df.rename(columns={
            'customerid': 'customer_id',
            'streamid': 'stream_id',
            'timesviewed': 'times_viewed',
            # Add other mappings as necessary
        }, inplace=True)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    print("Data loaded successfully!")
    return combined_df


def clean_data(df):
    """
    Clean the loaded DataFrame by dropping unnecessary columns,
    handling missing values, and ensuring consistency.
    """
    # Drop unnecessary columns
    df = df.drop(columns=["total_price", "StreamID", "TimesViewed"], errors="ignore")

    # Fill missing values
    df['customer_id'] = df['customer_id'].fillna(-1)  # Fill missing customer IDs with -1
    df['price'] = df['price'].fillna(0)  # Replace missing prices with 0
    df['stream_id'] = df['stream_id'].fillna("unknown")  # Replace missing stream IDs with 'unknown'
    df['times_viewed'] = df['times_viewed'].fillna(0)  # Replace missing times viewed with 0
    

    # Drop duplicates
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = df.columns.str.lower()

    print("Data cleaned successfully!")
    return df



def prepare_time_series(df):
    """
    Aggregate the cleaned data into a time-series format.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Time-series aggregated DataFrame.
    """
    # Combine year, month, and day into a single 'date' column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Aggregate data by date
    aggregated = df.groupby('date').agg(
        revenue=('price', 'sum'),
        unique_customers=('customer_id', lambda x: x.nunique()),
        total_views=('times_viewed', 'sum')
    ).reset_index()
    aggregated = aggregated.fillna(0)
    print("Data aggregated successfully!")
    return aggregated


def save_aggregated_data(df, file_path='aggregated_data.csv'):
    """
    Save the aggregated time-series data to a CSV file.

    Args:
        df (pd.DataFrame): Aggregated DataFrame.
        file_path (str): File path for saving.

    Returns:
        None
    """
    df.to_csv(file_path, index=False)
    print(f"Aggregated data saved to {file_path}!")


def train_test_split(df, split_ratio=0.8):
    """
    Split the time-series data into training and testing sets.

    Parameters:
        df (pd.DataFrame): The aggregated time-series data.
        split_ratio (float): The ratio of training to testing data (default is 0.8).

    Returns:
        tuple: A tuple containing the training set (pd.DataFrame) and the testing set (pd.DataFrame).
    """
    try:
        # Calculate the split index based on the ratio
        split_index = int(len(df) * split_ratio)

        # Split the data into training and testing sets
        train = df[:split_index]
        test = df[split_index:]

        print(f"Training set: {len(train)} samples")
        print(f"Testing set: {len(test)} samples")

        # Return the split data
        return train, test
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return None, None

def check_missing_values(df, label):
    """
    Check for missing values in a DataFrame and print results.

    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        label (str): A label to identify the stage of the check.
    """
    print(f"\n--- Missing Values Check: {label} ---")
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values detected:")
        print(missing[missing > 0])
    else:
        print("No missing values found.")
