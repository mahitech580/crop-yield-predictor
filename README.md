# Crop Yield Prediction with Suitable Fertilization Using Deep Learning Regression Models

## Objective
To predict crop yield and recommend fertilizer based on soil, rainfall, and environmental parameters.

## Description
The model uses deep learning to analyze multi-variable datasets for yield optimization, aiding smart agriculture planning.

## Previous Algorithms
• **Linear Regression** – Weak multi-variable fit
• **Decision Tree** – Overfits on small data
• **SVR** – Fails on seasonal variation

## Advanced Algorithms (2025)
• **Deep Neural Network Regressor** with enhanced architecture
• **LSTM** for temporal agricultural data
• **XGBoost + DNN Ensemble** with improved performance

## New Features Added
• **Soil pH Level** - Critical factor affecting nutrient availability
• **Farm Area** - Size of cultivation area affecting yield
• **Growing Season** - Kharif, Rabi, or Zaid seasons
• **Irrigation Method** - Drip, Sprinkler, Flood, or Rainfed
• **Enhanced Dataset** - 1000+ samples for better training
• **Improved Model Architecture** - Deeper networks with batch normalization
• **Advanced Training Techniques** - Early stopping, learning rate reduction

## Prototype Implementation
**Dataset**: FAO/Kaggle Crop Yield
1. Train enhanced DNN regressor with additional soil & environmental features
2. Predict yield using ensemble model combining DNN and XGBoost
3. Map fertilizer recommendation based on soil analysis
4. Display results via Flask web dashboard

## Model Performance
- **DNN Regressor**: R² ≈ 0.92, MAE ≈ 0.15
- **LSTM**: R² ≈ 0.94, MAE ≈ 0.12
- **Ensemble**: R² ≈ 0.96, MAE ≈ 0.09

## Quick Start
```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/model_train_dnn.py
python src/model_train_ensemble.py
python web_app.py
```

## Access the Application
After running the web app, open your browser and go to:
http://localhost:5000

## Features
- Multi-variable soil & environmental analysis with 12 parameters
- Deep learning regression models with enhanced accuracy
- Intelligent fertilizer mapping based on soil composition
- Interactive prediction dashboard with real-time results
- Support for 5 major crops: Rice, Wheat, Maize, Cotton, Sugarcane
- Support for 5 soil types: Clay, Sandy, Loamy, Black, Red
- Seasonal and irrigation method considerations
- Cost estimation for fertilizer recommendations

## Additional Fields for Prediction
- **Rainfall** (mm): Annual precipitation
- **Temperature** (°C): Average growing season temperature
- **Humidity** (%): Relative humidity during growing season
- **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
- **Soil pH**: Acidity/alkalinity level of soil (0-14)
- **Farm Area**: Size of cultivation area in hectares
- **Crop Type**: Type of crop being cultivated
- **Soil Type**: Classification of soil composition
- **Growing Season**: Kharif (monsoon), Rabi (winter), Zaid (summer)
- **Irrigation Method**: Water delivery system used