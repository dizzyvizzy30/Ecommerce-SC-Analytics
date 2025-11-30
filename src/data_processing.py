"""
Data Processing Module
Handles data cleaning, missing value imputation, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Class for processing and cleaning e-commerce datasets.
    """

    def __init__(self, data_dict):
        """
        Initialize DataProcessor with data dictionary.

        Parameters:
        -----------
        data_dict : dict
            Dictionary containing dataframes (customers, products, orders, order_items, payments)
        """
        self.data = data_dict.copy()
        self.processed_data = {}
        self.processing_log = []

    def log_step(self, message):
        """Log processing step."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(log_entry)

    def handle_missing_values_products(self):
        """
        Handle missing values in products dataset.
        """
        self.log_step("Processing products dataset...")
        products = self.data['products'].copy()

        initial_missing = products.isnull().sum().sum()

        # Fill missing product categories with 'Unknown'
        if 'product_category_name' in products.columns:
            missing_cat = products['product_category_name'].isnull().sum()
            if missing_cat > 0:
                products['product_category_name'].fillna('unknown_category', inplace=True)
                self.log_step(f"  Filled {missing_cat} missing product categories")

        # Fill missing dimensions with median values
        dimension_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        for col in dimension_cols:
            if col in products.columns:
                missing_count = products[col].isnull().sum()
                if missing_count > 0:
                    median_value = products[col].median()
                    products[col].fillna(median_value, inplace=True)
                    self.log_step(f"  Filled {missing_count} missing {col} with median: {median_value:.2f}")

        final_missing = products.isnull().sum().sum()
        self.log_step(f"  Products: {initial_missing} → {final_missing} missing values")

        self.processed_data['products'] = products
        return products

    def handle_missing_values_orders(self):
        """
        Handle missing values in orders dataset.
        """
        self.log_step("Processing orders dataset...")
        orders = self.data['orders'].copy()

        initial_missing = orders.isnull().sum().sum()

        # Convert datetime columns
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at',
                        'order_delivered_timestamp', 'order_estimated_delivery_date']

        for col in datetime_cols:
            if col in orders.columns:
                orders[col] = pd.to_datetime(orders[col], errors='coerce')

        # Fill missing approval dates with purchase timestamp (assume immediate approval)
        if 'order_approved_at' in orders.columns:
            missing_approved = orders['order_approved_at'].isnull().sum()
            if missing_approved > 0:
                orders['order_approved_at'].fillna(orders['order_purchase_timestamp'], inplace=True)
                self.log_step(f"  Filled {missing_approved} missing approval dates with purchase timestamp")

        # Mark orders with missing delivery timestamp as 'not_delivered' (if status column doesn't exist)
        if 'order_delivered_timestamp' in orders.columns:
            missing_delivery = orders['order_delivered_timestamp'].isnull().sum()
            if missing_delivery > 0:
                self.log_step(f"  {missing_delivery} orders have no delivery timestamp (in-transit/cancelled)")

        final_missing = orders.isnull().sum().sum()
        self.log_step(f"  Orders: {initial_missing} → {final_missing} missing values")

        self.processed_data['orders'] = orders
        return orders

    def handle_missing_values_customers(self):
        """
        Handle missing values in customers dataset.
        """
        self.log_step("Processing customers dataset...")
        customers = self.data['customers'].copy()

        initial_missing = customers.isnull().sum().sum()

        # Fill missing zip codes with mode
        if 'customer_zip_code_prefix' in customers.columns:
            missing_zip = customers['customer_zip_code_prefix'].isnull().sum()
            if missing_zip > 0:
                mode_zip = customers['customer_zip_code_prefix'].mode()[0]
                customers['customer_zip_code_prefix'].fillna(mode_zip, inplace=True)
                self.log_step(f"  Filled {missing_zip} missing zip codes with mode: {mode_zip}")

        # Fill missing cities/states with 'Unknown'
        for col in ['customer_city', 'customer_state']:
            if col in customers.columns:
                missing_count = customers[col].isnull().sum()
                if missing_count > 0:
                    customers[col].fillna('Unknown', inplace=True)
                    self.log_step(f"  Filled {missing_count} missing {col}")

        final_missing = customers.isnull().sum().sum()
        self.log_step(f"  Customers: {initial_missing} → {final_missing} missing values")

        self.processed_data['customers'] = customers
        return customers

    def handle_missing_values_order_items(self):
        """
        Handle missing values in order items dataset.
        """
        self.log_step("Processing order items dataset...")
        order_items = self.data['order_items'].copy()

        initial_missing = order_items.isnull().sum().sum()

        # Remove rows with missing critical values (price, order_id, product_id)
        critical_cols = ['order_id', 'product_id']
        missing_critical = order_items[critical_cols].isnull().any(axis=1).sum()

        if missing_critical > 0:
            order_items = order_items.dropna(subset=critical_cols)
            self.log_step(f"  Removed {missing_critical} rows with missing critical values")

        # Fill missing price/shipping with 0
        for col in ['price', 'shipping_charges']:
            if col in order_items.columns:
                missing_count = order_items[col].isnull().sum()
                if missing_count > 0:
                    order_items[col].fillna(0, inplace=True)
                    self.log_step(f"  Filled {missing_count} missing {col} with 0")

        final_missing = order_items.isnull().sum().sum()
        self.log_step(f"  Order Items: {initial_missing} → {final_missing} missing values")

        self.processed_data['order_items'] = order_items
        return order_items

    def handle_missing_values_payments(self):
        """
        Handle missing values in payments dataset.
        """
        self.log_step("Processing payments dataset...")
        payments = self.data['payments'].copy()

        initial_missing = payments.isnull().sum().sum()

        # Fill missing payment type with 'unknown'
        if 'payment_type' in payments.columns:
            missing_type = payments['payment_type'].isnull().sum()
            if missing_type > 0:
                payments['payment_type'].fillna('unknown', inplace=True)
                self.log_step(f"  Filled {missing_type} missing payment types")

        # Fill missing payment value with 0
        if 'payment_value' in payments.columns:
            missing_value = payments['payment_value'].isnull().sum()
            if missing_value > 0:
                payments['payment_value'].fillna(0, inplace=True)
                self.log_step(f"  Filled {missing_value} missing payment values with 0")

        # Fill missing installments with 1
        if 'payment_installments' in payments.columns:
            missing_inst = payments['payment_installments'].isnull().sum()
            if missing_inst > 0:
                payments['payment_installments'].fillna(1, inplace=True)
                self.log_step(f"  Filled {missing_inst} missing installments with 1")

        final_missing = payments.isnull().sum().sum()
        self.log_step(f"  Payments: {initial_missing} → {final_missing} missing values")

        self.processed_data['payments'] = payments
        return payments

    def remove_duplicates(self):
        """
        Remove duplicate rows from all datasets.
        """
        self.log_step("Removing duplicates...")

        for name, df in self.processed_data.items():
            initial_shape = df.shape[0]

            # For products, keep duplicates as they may represent same product in different orders
            if name == 'products':
                self.log_step(f"  {name}: Keeping duplicates (products can appear in multiple orders)")
                continue

            # For other datasets, remove duplicates
            df_dedup = df.drop_duplicates()
            duplicates_removed = initial_shape - df_dedup.shape[0]

            if duplicates_removed > 0:
                self.processed_data[name] = df_dedup
                self.log_step(f"  {name}: Removed {duplicates_removed} duplicates")
            else:
                self.log_step(f"  {name}: No duplicates found")

    def process_all(self):
        """
        Execute all processing steps.

        Returns:
        --------
        dict : Dictionary containing processed dataframes
        """
        self.log_step("="*60)
        self.log_step("STARTING DATA PROCESSING PIPELINE")
        self.log_step("="*60)

        # Handle missing values for each dataset
        self.handle_missing_values_customers()
        self.handle_missing_values_products()
        self.handle_missing_values_orders()
        self.handle_missing_values_order_items()
        self.handle_missing_values_payments()

        # Remove duplicates
        self.remove_duplicates()

        self.log_step("="*60)
        self.log_step("DATA PROCESSING COMPLETE")
        self.log_step("="*60)

        return self.processed_data

    def save_processed_data(self, output_path='data/processed'):
        """
        Save processed data to CSV files.

        Parameters:
        -----------
        output_path : str
            Path to save processed data
        """
        import os
        os.makedirs(output_path, exist_ok=True)

        self.log_step(f"\nSaving processed data to {output_path}...")

        file_mapping = {
            'customers': 'customers_processed.csv',
            'products': 'products_processed.csv',
            'orders': 'orders_processed.csv',
            'order_items': 'order_items_processed.csv',
            'payments': 'payments_processed.csv'
        }

        for key, filename in file_mapping.items():
            if key in self.processed_data:
                filepath = os.path.join(output_path, filename)
                self.processed_data[key].to_csv(filepath, index=False)
                self.log_step(f"  Saved {filename} - Shape: {self.processed_data[key].shape}")

        # Save processing log
        log_file = os.path.join(output_path, 'processing_log.txt')
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.processing_log))
        self.log_step(f"  Saved processing log: {log_file}")

    def get_summary(self):
        """
        Get summary of processed data.

        Returns:
        --------
        pd.DataFrame : Summary dataframe
        """
        summary_data = []

        for name, df in self.processed_data.items():
            summary_data.append({
                'Dataset': name,
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Missing Values': df.isnull().sum().sum(),
                'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df


def process_data(data_dict, save_output=True, output_path='data/processed'):
    """
    Convenience function to process data.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing raw dataframes
    save_output : bool
        Whether to save processed data
    output_path : str
        Path to save processed data

    Returns:
    --------
    dict : Dictionary containing processed dataframes
    """
    processor = DataProcessor(data_dict)
    processed_data = processor.process_all()

    if save_output:
        processor.save_processed_data(output_path)

    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(processor.get_summary().to_string(index=False))

    return processed_data
