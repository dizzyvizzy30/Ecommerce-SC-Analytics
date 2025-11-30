"""
Feature Engineering Module
Creates new features for machine learning models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Class for creating features from e-commerce datasets.
    """

    def __init__(self, data_dict):
        """
        Initialize FeatureEngineer with processed data dictionary.

        Parameters:
        -----------
        data_dict : dict
            Dictionary containing processed dataframes
        """
        self.data = data_dict
        self.features = None

    def create_time_features(self, orders):
        """
        Create time-based features from orders data.

        Parameters:
        -----------
        orders : pd.DataFrame
            Orders dataframe

        Returns:
        --------
        pd.DataFrame : Orders with time features
        """
        orders = orders.copy()

        # Convert to datetime if not already
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at',
                        'order_delivered_timestamp', 'order_estimated_delivery_date']

        for col in datetime_cols:
            if col in orders.columns and orders[col].dtype != 'datetime64[ns]':
                orders[col] = pd.to_datetime(orders[col], errors='coerce')

        # Extract temporal features from purchase timestamp
        if 'order_purchase_timestamp' in orders.columns:
            orders['purchase_year'] = orders['order_purchase_timestamp'].dt.year
            orders['purchase_month'] = orders['order_purchase_timestamp'].dt.month
            orders['purchase_day'] = orders['order_purchase_timestamp'].dt.day
            orders['purchase_dayofweek'] = orders['order_purchase_timestamp'].dt.dayofweek
            orders['purchase_hour'] = orders['order_purchase_timestamp'].dt.hour
            orders['purchase_quarter'] = orders['order_purchase_timestamp'].dt.quarter

            # Weekend flag
            orders['is_weekend'] = orders['purchase_dayofweek'].isin([5, 6]).astype(int)

        # Approval time (time from purchase to approval)
        if 'order_purchase_timestamp' in orders.columns and 'order_approved_at' in orders.columns:
            orders['approval_time_hours'] = (
                (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600
            )
            orders['approval_time_hours'] = orders['approval_time_hours'].clip(lower=0)

        # Delivery time (time from approval to delivery)
        if 'order_approved_at' in orders.columns and 'order_delivered_timestamp' in orders.columns:
            orders['delivery_time_days'] = (
                (orders['order_delivered_timestamp'] - orders['order_approved_at']).dt.total_seconds() / 86400
            )
            orders['delivery_time_days'] = orders['delivery_time_days'].clip(lower=0)

        # Expected delivery time (time from approval to estimated delivery)
        if 'order_approved_at' in orders.columns and 'order_estimated_delivery_date' in orders.columns:
            orders['expected_delivery_days'] = (
                (orders['order_estimated_delivery_date'] - orders['order_approved_at']).dt.total_seconds() / 86400
            )

        # Delivery delay (actual vs estimated)
        if 'delivery_time_days' in orders.columns and 'expected_delivery_days' in orders.columns:
            orders['delivery_delay_days'] = orders['delivery_time_days'] - orders['expected_delivery_days']
            orders['is_delayed'] = (orders['delivery_delay_days'] > 0).astype(int)
            orders['is_early'] = (orders['delivery_delay_days'] < 0).astype(int)
            orders['is_on_time'] = (orders['delivery_delay_days'].abs() <= 1).astype(int)

        # Processing time (purchase to delivery)
        if 'order_purchase_timestamp' in orders.columns and 'order_delivered_timestamp' in orders.columns:
            orders['total_processing_days'] = (
                (orders['order_delivered_timestamp'] - orders['order_purchase_timestamp']).dt.total_seconds() / 86400
            )

        return orders

    def create_order_value_features(self, order_items, payments):
        """
        Create order value and payment features.

        Parameters:
        -----------
        order_items : pd.DataFrame
            Order items dataframe
        payments : pd.DataFrame
            Payments dataframe

        Returns:
        --------
        pd.DataFrame : Order-level value features
        """
        # Aggregate order items
        order_agg = order_items.groupby('order_id').agg({
            'product_id': 'count',  # Number of items
            'price': ['sum', 'mean', 'std', 'min', 'max'],
            'shipping_charges': ['sum', 'mean']
        }).reset_index()

        order_agg.columns = [
            'order_id', 'num_items', 'total_price', 'avg_item_price',
            'std_item_price', 'min_item_price', 'max_item_price',
            'total_shipping', 'avg_shipping'
        ]

        # Fill NaN std with 0 (single item orders)
        order_agg['std_item_price'].fillna(0, inplace=True)

        # Price range
        order_agg['price_range'] = order_agg['max_item_price'] - order_agg['min_item_price']

        # Total order value
        order_agg['order_value'] = order_agg['total_price'] + order_agg['total_shipping']

        # Aggregate payments
        payment_agg = payments.groupby('order_id').agg({
            'payment_value': 'sum',
            'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
            'payment_installments': 'max'
        }).reset_index()

        payment_agg.columns = ['order_id', 'total_payment', 'primary_payment_type', 'max_installments']

        # Merge order and payment features
        order_features = order_agg.merge(payment_agg, on='order_id', how='left')

        # Payment difference (should be close to 0 if data is consistent)
        order_features['payment_diff'] = order_features['total_payment'] - order_features['order_value']

        # Using installments flag
        order_features['uses_installments'] = (order_features['max_installments'] > 1).astype(int)

        return order_features

    def create_product_features(self, products, order_items):
        """
        Create product-related features.

        Parameters:
        -----------
        products : pd.DataFrame
            Products dataframe
        order_items : pd.DataFrame
            Order items dataframe

        Returns:
        --------
        pd.DataFrame : Products with additional features
        """
        products = products.copy()

        # Product volume (cm³)
        if all(col in products.columns for col in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
            products['product_volume_cm3'] = (
                products['product_length_cm'] *
                products['product_height_cm'] *
                products['product_width_cm']
            )

        # Weight-to-volume ratio
        if 'product_weight_g' in products.columns and 'product_volume_cm3' in products.columns:
            products['weight_volume_ratio'] = products['product_weight_g'] / (products['product_volume_cm3'] + 1)

        # Product popularity (number of times ordered)
        if 'product_id' in order_items.columns:
            product_popularity = order_items.groupby('product_id').size().reset_index(name='times_ordered')
            products = products.merge(product_popularity, on='product_id', how='left')
            products['times_ordered'].fillna(0, inplace=True)

        # Category encoding (one-hot encoding would be done in modeling)
        if 'product_category_name' in products.columns:
            products['has_category'] = (~products['product_category_name'].isin(['unknown_category', 'Unknown'])).astype(int)

        return products

    def create_customer_features(self, customers, orders, order_items):
        """
        Create customer-level aggregated features.

        Parameters:
        -----------
        customers : pd.DataFrame
            Customers dataframe
        orders : pd.DataFrame
            Orders dataframe
        order_items : pd.DataFrame
            Order items dataframe

        Returns:
        --------
        pd.DataFrame : Customer features
        """
        # Merge orders with order items to get order values
        order_values = order_items.groupby('order_id').agg({
            'price': 'sum',
            'product_id': 'count'
        }).reset_index()
        order_values.columns = ['order_id', 'order_total', 'num_items']

        orders_with_values = orders.merge(order_values, on='order_id', how='left')

        # Customer-level aggregations
        customer_agg = orders_with_values.groupby('customer_id').agg({
            'order_id': 'count',  # Number of orders
            'order_total': ['sum', 'mean', 'max'],
            'num_items': ['sum', 'mean']
        }).reset_index()

        customer_agg.columns = [
            'customer_id', 'total_orders', 'total_spent', 'avg_order_value',
            'max_order_value', 'total_items_purchased', 'avg_items_per_order'
        ]

        # Customer segment based on total orders
        customer_agg['customer_segment'] = pd.cut(
            customer_agg['total_orders'],
            bins=[0, 1, 3, 10, float('inf')],
            labels=['one_time', 'occasional', 'regular', 'frequent']
        )

        # Merge with customer location data
        customer_features = customers.merge(customer_agg, on='customer_id', how='left')

        # Fill NaN for customers with no orders
        numeric_cols = ['total_orders', 'total_spent', 'avg_order_value',
                       'max_order_value', 'total_items_purchased', 'avg_items_per_order']
        for col in numeric_cols:
            if col in customer_features.columns:
                customer_features[col].fillna(0, inplace=True)

        return customer_features

    def create_seller_features(self, order_items):
        """
        Create seller-level features.

        Parameters:
        -----------
        order_items : pd.DataFrame
            Order items dataframe

        Returns:
        --------
        pd.DataFrame : Seller features
        """
        seller_agg = order_items.groupby('seller_id').agg({
            'order_id': 'count',  # Number of sales
            'price': ['sum', 'mean'],
            'product_id': 'nunique'  # Number of unique products
        }).reset_index()

        seller_agg.columns = [
            'seller_id', 'total_sales', 'total_revenue',
            'avg_sale_price', 'unique_products'
        ]

        # Seller performance tier
        seller_agg['seller_tier'] = pd.cut(
            seller_agg['total_sales'],
            bins=[0, 10, 50, 200, float('inf')],
            labels=['bronze', 'silver', 'gold', 'platinum']
        )

        return seller_agg

    def build_master_dataset(self):
        """
        Build a master dataset with all features for modeling.

        Returns:
        --------
        pd.DataFrame : Master dataset with all features
        """
        print("Building master dataset...")

        # Create all features
        print("  Creating time features...")
        orders_features = self.create_time_features(self.data['orders'])

        print("  Creating order value features...")
        order_value_features = self.create_order_value_features(
            self.data['order_items'],
            self.data['payments']
        )

        print("  Creating product features...")
        product_features = self.create_product_features(
            self.data['products'],
            self.data['order_items']
        )

        print("  Creating customer features...")
        customer_features = self.create_customer_features(
            self.data['customers'],
            self.data['orders'],
            self.data['order_items']
        )

        print("  Creating seller features...")
        seller_features = self.create_seller_features(self.data['order_items'])

        # Merge all features
        print("  Merging all features...")

        # Start with orders
        master = orders_features.copy()

        # Add order value features
        master = master.merge(order_value_features, on='order_id', how='left')

        # Add customer features
        master = master.merge(
            customer_features[['customer_id', 'customer_state', 'total_orders',
                              'total_spent', 'avg_order_value', 'customer_segment']],
            on='customer_id',
            how='left'
        )

        # Add order items and product info
        order_items_with_products = self.data['order_items'].merge(
            product_features,
            on='product_id',
            how='left'
        )

        # Add seller info
        order_items_with_products = order_items_with_products.merge(
            seller_features[['seller_id', 'seller_tier']],
            on='seller_id',
            how='left'
        )

        # Aggregate product features at order level
        product_agg = order_items_with_products.groupby('order_id').agg({
            'product_volume_cm3': 'mean',
            'product_weight_g': 'mean',
            'times_ordered': 'mean',
            'product_category_name': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()

        product_agg.columns = [
            'order_id', 'avg_product_volume', 'avg_product_weight',
            'avg_product_popularity', 'primary_category'
        ]

        master = master.merge(product_agg, on='order_id', how='left')

        self.features = master
        print(f"Master dataset created: {master.shape}")
        print(f"Features: {master.columns.tolist()}")

        return master

    def save_features(self, output_path='data/processed/master_features.csv'):
        """
        Save engineered features to CSV.

        Parameters:
        -----------
        output_path : str
            Path to save features
        """
        if self.features is None:
            raise ValueError("No features to save. Run build_master_dataset() first.")

        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.features.to_csv(output_path, index=False)
        print(f"Features saved: {output_path}")
        print(f"Shape: {self.features.shape}")


def engineer_features(data_dict, save_output=True, output_path='data/processed/master_features.csv'):
    """
    Convenience function to engineer features.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing processed dataframes
    save_output : bool
        Whether to save features
    output_path : str
        Path to save features

    Returns:
    --------
    pd.DataFrame : Master dataset with engineered features
    """
    engineer = FeatureEngineer(data_dict)
    features = engineer.build_master_dataset()

    if save_output:
        engineer.save_features(output_path)

    return features
