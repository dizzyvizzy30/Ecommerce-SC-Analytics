import pandas as pd
import os

class DataLoader:
    """Utility class to load and manage e-commerce dataset"""

    def __init__(self, data_path="data/raw"):
        self.data_path = data_path
        self.test_path = os.path.join(data_path, "test")
        self.train_path = os.path.join(data_path, "train")
        self.data = {}

    def list_files(self):
        """List all files in the test and train directories"""
        for folder in [self.test_path, self.train_path]:
            print(f"\nFiles in {folder}:")
            for file in os.listdir(folder):
                print(f"  - {file}")

    def load_csv_from_folder(self, folder_path):
        """Load all CSV files from a specific folder"""
        dataframes = {}
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                dataframes[file] = df
                print(f"✓ Loaded {file}: {df.shape}")
        return dataframes

    def load_all_data(self):
        """Load all data from test and train folders"""
        print("\nLoading test data...")
        self.data['test'] = self.load_csv_from_folder(self.test_path)
        print("\nLoading train data...")
        self.data['train'] = self.load_csv_from_folder(self.train_path)
        return self.data

    def summary(self):
        """Print summary of loaded data"""
        print("\n=== Dataset Summary ===")
        for category, datasets in self.data.items():
            print(f"\nCategory: {category}")
            for filename, df in datasets.items():
                print(f"  {filename}:")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Usage
if __name__ == "__main__":
    loader = DataLoader()
    loader.list_files()
    loader.load_all_data()
    loader.summary()