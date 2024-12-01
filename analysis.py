import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from prophet import Prophet

def analyze_seasonality(df, output_dir, show_plots=True):
    """
    Analyze seasonality and trends using Prophet.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'date' and 'revenue' columns.
        output_dir (str): Directory to save plots.
        show_plots (bool): Whether to show or save the plot.
    """
    try:
        print("\nAnalyzing Seasonality...")
        prophet_train = df.rename(columns={'date': 'ds', 'revenue': 'y'})
        prophet_model = Prophet()
        prophet_model.fit(prophet_train)
        forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=0))

        # Debugging: Print the forecast to confirm it's generated
        print("Forecast generated successfully for seasonality analysis.")

        # Save seasonality components plot
        fig = prophet_model.plot_components(forecast)
        file_path = os.path.join(output_dir, "seasonality_components.png")
        fig.savefig(file_path)
        print(f"Seasonality components plot saved to {file_path}")
        if show_plots:
            plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Error during seasonality analysis: {e}")


def analyze_customer_impact(df, output_dir, show_plots=True):
    """
    Analyze the relationship between unique customers and revenue.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'unique_customers' and 'revenue' columns.
        output_dir (str): Directory to save plots.
        show_plots (bool): Whether to show or save the plot.
    """
    try:
        correlation = np.corrcoef(df['unique_customers'], df['revenue'])[0, 1]
        print(f"Correlation between unique customers and revenue: {correlation}")

        plt.figure(figsize=(10, 5))
        plt.scatter(df['unique_customers'], df['revenue'], alpha=0.6)
        plt.title('Unique Customers vs Revenue')
        plt.xlabel('Unique Customers')
        plt.ylabel('Revenue')
        plt.grid(True)

        file_path = os.path.join(output_dir, "customer_vs_revenue.png")
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
        if show_plots:
            plt.show()
        plt.close()

    except Exception as e:
        print(f"Error during customer impact analysis: {e}")


def analyze_engagement(df, output_dir, show_plots=True):
    """
    Analyze the relationship between total views and revenue over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'date', 'total_views', and 'revenue' columns.
        output_dir (str): Directory to save plots.
        show_plots (bool): Whether to show or save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue', color='blue')
    ax1.plot(df['date'], df['revenue'], label='Revenue', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Views', color='orange')
    ax2.plot(df['date'], df['total_views'], label='Total Views', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Engagement vs Revenue')

    file_path = os.path.join(output_dir, "engagement_vs_revenue.png")
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    if show_plots:
        plt.show()
    plt.close()


def analyze_relationships(df):
    """
    Analyze the relationships between features using linear regression.

    Parameters:
        df (DataFrame): DataFrame containing 'unique_customers', 'total_views', and 'revenue' columns.
    """
    model = LinearRegression()
    X = df[['unique_customers', 'total_views']]
    y = df['revenue']
    model.fit(X, y)

    print("Linear Regression Results:")
    print(f"Coefficient for 'unique_customers': {model.coef_[0]:.2f}")
    print(f"Coefficient for 'total_views': {model.coef_[1]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")


def analyze_relationships_with_features(df):
    """
    Analyze relationships using derived features and linear regression.

    Parameters:
        df (DataFrame): DataFrame containing 'unique_customers', 'total_views', 
                        'revenue_per_customer', 'revenue_lag_1', and 'revenue'.
    """
    model = LinearRegression()
    features = ['unique_customers', 'total_views', 'revenue_per_customer', 'revenue_lag_1']
    df = df.dropna()  # Drop rows with missing values caused by lag features
    X = df[features]
    y = df['revenue']
    model.fit(X, y)

    print("Linear Regression with Derived Features:")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
