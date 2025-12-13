import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

def load_kaggle_dataset():
    """Load the Kaggle crop yield prediction dataset"""
    print("Loading Kaggle dataset...")
    
    try:
        # Download the dataset first
        path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")
        print(f"Dataset downloaded to: {path}")
        
        # Find CSV files in the downloaded path
        import glob
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        print(f"Found CSV files: {csv_files}")
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
        else:
            raise FileNotFoundError("No CSV files found in the dataset")
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 records:")
        print(df.head())
        print("\nDataset columns:")
        print(df.columns.tolist())
        print("\nDataset info:")
        print(df.info())
        
        # Save to raw data folder
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/kaggle_crop_yield.csv', index=False)
        print("\nDataset saved to data/raw/kaggle_crop_yield.csv")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    df = load_kaggle_dataset()