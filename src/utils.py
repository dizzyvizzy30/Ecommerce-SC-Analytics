"""
Utility functions for E-commerce Supply Chain Analytics
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path='data/raw', split='train'):
    """
    Load all CSV files from specified data split (train or test).

    Parameters:
    -----------
    data_path : str
        Base path to data directory
    split : str
        Data split to load ('train' or 'test')

    Returns:
    --------
    dict : Dictionary containing all dataframes
    """
    folder_path = os.path.join(data_path, split)
    dataframes = {}

    file_mapping = {
        'df_Customers.csv': 'customers',
        'df_Products.csv': 'products',
        'df_Orders.csv': 'orders',
        'df_OrderItems.csv': 'order_items',
        'df_Payments.csv': 'payments'
    }

    for filename, key in file_mapping.items():
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            dataframes[key] = pd.read_csv(file_path)
            print(f"Loaded {key}: {dataframes[key].shape}")
        else:
            print(f"Warning: {filename} not found in {folder_path}")

    return dataframes


def convert_datetime_columns(df, datetime_cols):
    """
    Convert specified columns to datetime format.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_cols : list
        List of column names to convert

    Returns:
    --------
    pd.DataFrame : DataFrame with converted datetime columns
    """
    df = df.copy()
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def calculate_missing_percentage(df):
    """
    Calculate percentage of missing values for each column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame : DataFrame with missing value statistics
    """
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
    return missing_stats


def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers

    Returns:
    --------
    pd.Series : Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def save_figure(fig, filename, folder='results/figures'):
    """
    Save matplotlib figure to specified folder.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename for saved figure
    folder : str
        Folder path to save figure
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filepath}")


def plot_missing_values(df, title='Missing Values Distribution'):
    """
    Create visualization for missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    title : str
        Plot title

    Returns:
    --------
    matplotlib.figure.Figure : Figure object
    """
    missing_stats = calculate_missing_percentage(df)

    if len(missing_stats) == 0:
        print("No missing values found")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=missing_stats, x='missing_percentage', y='column', ax=ax, palette='viridis')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_ylabel('Column')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def get_numeric_columns(df):
    """
    Get list of numeric columns from dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    list : List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """
    Get list of categorical columns from dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    list : List of categorical column names
    """
    return df.select_dtypes(include=['object']).columns.tolist()


def merge_datasets(orders, order_items, products, customers, payments):
    """
    Merge all datasets into a single dataframe.

    Parameters:
    -----------
    orders : pd.DataFrame
        Orders dataframe
    order_items : pd.DataFrame
        Order items dataframe
    products : pd.DataFrame
        Products dataframe
    customers : pd.DataFrame
        Customers dataframe
    payments : pd.DataFrame
        Payments dataframe

    Returns:
    --------
    pd.DataFrame : Merged dataframe
    """
    # Merge orders with customers
    merged = orders.merge(customers, on='customer_id', how='left')

    # Merge with order items
    merged = merged.merge(order_items, on='order_id', how='left')

    # Merge with products
    merged = merged.merge(products, on='product_id', how='left')

    # Merge with payments
    merged = merged.merge(payments, on='order_id', how='left')

    print(f"Merged dataset shape: {merged.shape}")
    return merged


def create_summary_statistics(df, output_path='results/summary_stats.csv'):
    """
    Create and save summary statistics for numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_path : str
        Path to save summary statistics

    Returns:
    --------
    pd.DataFrame : Summary statistics dataframe
    """
    numeric_cols = get_numeric_columns(df)
    summary = df[numeric_cols].describe()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path)
    print(f"Summary statistics saved: {output_path}")

    return summary


def print_data_info(data_dict):
    """
    Print information about all dataframes in dictionary.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing dataframes
    """
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)

    for name, df in data_dict.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"  Missing values: {missing}")


def validate_referential_integrity(orders, order_items, customers, products):
    """
    Validate referential integrity between tables.

    Parameters:
    -----------
    orders : pd.DataFrame
        Orders dataframe
    order_items : pd.DataFrame
        Order items dataframe
    customers : pd.DataFrame
        Customers dataframe
    products : pd.DataFrame
        Products dataframe

    Returns:
    --------
    dict : Dictionary with validation results
    """
    results = {}

    # Check customers in orders
    customers_in_orders = orders['customer_id'].unique()
    valid_customers = set(customers_in_orders).issubset(set(customers['customer_id'].unique()))
    results['customers_integrity'] = valid_customers

    # Check orders in order_items
    orders_in_items = order_items['order_id'].unique()
    valid_orders = set(orders_in_items).issubset(set(orders['order_id'].unique()))
    results['orders_integrity'] = valid_orders

    # Check products in order_items
    products_in_items = order_items['product_id'].unique()
    valid_products = set(products_in_items).issubset(set(products['product_id'].unique()))
    results['products_integrity'] = valid_products

    print("\nReferential Integrity Check:")
    for key, value in results.items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")

    return results
