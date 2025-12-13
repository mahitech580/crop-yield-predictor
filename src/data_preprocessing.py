import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def create_sample_data():
    """Generate sample crop yield data for demonstration with additional features"""
    np.random.seed(42)
    n_samples = 2000  # Increased sample size for better training
    
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
    soil_types = ['Clay', 'Sandy', 'Loamy', 'Black', 'Red']
    
    # Generate base data with more realistic distributions
    data = {
        'Crop': np.random.choice(crops, n_samples, p=[0.25, 0.2, 0.2, 0.15, 0.2]),
        'Soil_Type': np.random.choice(soil_types, n_samples, p=[0.2, 0.15, 0.3, 0.2, 0.15]),
        'Rainfall': np.random.normal(800, 200, n_samples),
        'Temperature': np.random.normal(25, 5, n_samples),
        'Humidity': np.random.normal(65, 15, n_samples),
        'N': np.random.normal(50, 15, n_samples),
        'P': np.random.normal(30, 10, n_samples),
        'K': np.random.normal(40, 12, n_samples),
        'pH': np.random.normal(6.5, 0.8, n_samples),  # More realistic pH distribution
        'Area': np.random.lognormal(1.6, 0.5, n_samples),    # Log-normal distribution for area
        'Season': np.random.choice(['Kharif', 'Rabi', 'Zaid'], n_samples, p=[0.5, 0.3, 0.2]),  # Realistic season distribution
        'Irrigation': np.random.choice(['Drip', 'Sprinkler', 'Flood', 'Rainfed'], n_samples, p=[0.1, 0.2, 0.5, 0.2]),  # Realistic irrigation distribution
    }
    
    # Generate yield based on features with more complex relationships
    df = pd.DataFrame(data)
    
    # More realistic yield calculation considering interactions and non-linear relationships
    df['Yield'] = (
        0.003 * df['Rainfall'] + 
        0.15 * df['Temperature'] + 
        0.08 * df['Humidity'] + 
        0.25 * df['N'] + 
        0.2 * df['P'] + 
        0.15 * df['K'] +
        0.4 * df['pH'] +  # Increased pH importance
        0.05 * np.log(df['Area'] + 1) +  # Logarithmic effect of area
        np.random.normal(0, 0.8, n_samples)  # Reduced noise for better signal
    )
    
    # Adjust yield based on categorical factors with more realistic multipliers
    season_multiplier = {'Kharif': 1.2, 'Rabi': 1.0, 'Zaid': 0.8}
    irrigation_multiplier = {'Drip': 1.3, 'Sprinkler': 1.15, 'Flood': 1.0, 'Rainfed': 0.85}
    
    # Crop-specific multipliers for more realism
    crop_multiplier = {'Rice': 1.1, 'Wheat': 1.0, 'Maize': 1.2, 'Cotton': 0.9, 'Sugarcane': 1.4}
    soil_multiplier = {'Clay': 0.9, 'Sandy': 0.8, 'Loamy': 1.1, 'Black': 1.05, 'Red': 0.95}
    
    df['Yield'] = (df['Yield'] * 
                   df['Season'].map(season_multiplier) * 
                   df['Irrigation'].map(irrigation_multiplier) *
                   df['Crop'].map(crop_multiplier) *
                   df['Soil_Type'].map(soil_multiplier))
    
    # Ensure positive yields with more realistic minimum values
    df['Yield'] = np.maximum(df['Yield'], 0.8)
    
    # Cap unrealistic maximum values
    df['Yield'] = np.minimum(df['Yield'], 15.0)
    
    return df

def preprocess_data():
    """Clean and preprocess crop yield data with additional features"""
    # Try to load datasets in order of preference
    enhanced_path = 'data/raw/crop_yield_enhanced.csv'
    kaggle_path = 'data/raw/crop_yield_kaggle.csv'
    raw_path = 'data/raw/crop_yield.csv'
    
    if os.path.exists(enhanced_path):
        print("Loading enhanced dataset with additional features...")
        df = pd.read_csv(enhanced_path)
        print(f"Loaded enhanced dataset with {len(df)} samples")
    elif os.path.exists(kaggle_path):
        print("Loading Kaggle dataset...")
        df = pd.read_csv(kaggle_path)
        print(f"Loaded Kaggle dataset with {len(df)} samples")
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        print("Loaded existing dataset")
    else:
        os.makedirs('data/raw', exist_ok=True)
        df = create_sample_data()
        df.to_csv(raw_path, index=False)
        print("Sample data created")
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_season = LabelEncoder()
    le_irrigation = LabelEncoder()
    
    df['Crop_Encoded'] = le_crop.fit_transform(df['Crop'])
    df['Soil_Encoded'] = le_soil.fit_transform(df['Soil_Type'])
    
    # Add encoding for new categorical features if they exist
    if 'Season' in df.columns:
        df['Season_Encoded'] = le_season.fit_transform(df['Season'])
    else:
        df['Season_Encoded'] = 0  # Default value
        
    if 'Irrigation' in df.columns:
        df['Irrigation_Encoded'] = le_irrigation.fit_transform(df['Irrigation'])
    else:
        df['Irrigation_Encoded'] = 0  # Default value
    
    # Select features for modeling (including new features)
    base_features = ['Rainfall', 'Temperature', 'Humidity', 'N', 'P', 'K']
    
    # Check if new features exist in the dataset
    additional_features = []
    if 'pH' in df.columns:
        additional_features.append('pH')
    if 'Area' in df.columns:
        additional_features.append('Area')
    if 'Season_Encoded' in df.columns:
        additional_features.append('Season_Encoded')
    if 'Irrigation_Encoded' in df.columns:
        additional_features.append('Irrigation_Encoded')
    
    # Include encoded categorical variables
    categorical_features = ['Crop_Encoded', 'Soil_Encoded']
    
    all_features = base_features + additional_features + categorical_features
    X = df[all_features]
    y = df['Yield']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=all_features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    # Save encoders and scaler
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_crop, 'models/crop_encoder.pkl')
    joblib.dump(le_soil, 'models/soil_encoder.pkl')
    
    # Save new encoders if they exist
    if 'Season' in df.columns:
        joblib.dump(le_season, 'models/season_encoder.pkl')
    if 'Irrigation' in df.columns:
        joblib.dump(le_irrigation, 'models/irrigation_encoder.pkl')
    
    print(f"Data preprocessed: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Dataset shape: {df.shape}")
    print(f"Features used: {all_features}")
    print(f"Yield range: {y.min():.2f} - {y.max():.2f}")
    print(f"Mean yield: {y.mean():.2f}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()