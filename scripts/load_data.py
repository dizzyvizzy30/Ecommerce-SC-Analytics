import kagglehub
import pandas as pd
import os
from pathlib import Path

class DataLoader:
    """Utility class to load and manage e-commerce dataset"""
    
    def __init__(self):
        self.dataset_path = None
        self.data = {}
    
    def download_dataset(self):
        """Download the dataset from Kaggle"""
        try:
            self.dataset_path = kagglehub.dataset_download("bytadit/ecommerce-order-dataset")
            print(f"✓ Dataset downloaded to: {self.dataset_path}")
            return self.dataset_path
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            return None
    
    def list_files(self):
        """List all files in the dataset"""
        if not self.dataset_path:
            print("Dataset not downloaded yet. Call download_dataset() first.")
            return []
        
        files = os.listdir(self.dataset_path)
        print(f"\nAvailable files ({len(files)}):")
        for f in files:
            print(f"  - {f}")
        return files
    
    def load_all_csv(self):
        """Load all CSV files from the dataset"""
        if not self.dataset_path:
            self.download_dataset()
        
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(self.dataset_path, csv_file)
            df = pd.read_csv(file_path)
            self.data[csv_file] = df
            print(f"✓ Loaded {csv_file}: {df.shape}")
        
        return self.data
    
    def save_to_local(self, output_dir="data/raw"):
        """Save all loaded data to local directory"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for filename, df in self.data.items():
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"✓ Saved {filename} to {output_dir}/")
    
    def summary(self):
        """Print summary of loaded data"""
        print("\n=== Dataset Summary ===")
        for filename, df in self.data.items():
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Usage
if __name__ == "__main__":
    loader = DataLoader()
    loader.download_dataset()
    loader.list_files()
    loader.load_all_csv()
    loader.save_to_local()
    loader.summary()