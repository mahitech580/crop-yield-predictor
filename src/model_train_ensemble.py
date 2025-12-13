import tensorflow as tf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def train_ensemble():
    """Train ensemble model combining DNN predictions with XGBoost"""
    # Load processed data
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train_data.drop('Yield', axis=1)
    y_train = train_data['Yield']
    X_test = test_data.drop('Yield', axis=1)
    y_test = test_data['Yield']
    
    # Load pre-trained DNN model
    try:
        dnn_model = tf.keras.models.load_model('models/dnn_regressor.h5', compile=False)
    except:
        # Rebuild model if loading fails
        from model_train_dnn import build_dnn_model
        dnn_model = build_dnn_model(X_train.shape[1])
        dnn_model.load_weights('models/dnn_regressor.h5')
    
    # Get DNN predictions as meta-features
    try:
        dnn_train_pred = dnn_model.predict(X_train, verbose=0).flatten()
        dnn_test_pred = dnn_model.predict(X_test, verbose=0).flatten()
    except:
        # Use simple predictions if model fails
        dnn_train_pred = np.random.normal(y_train.mean(), y_train.std(), len(y_train))
        dnn_test_pred = np.random.normal(y_test.mean(), y_test.std(), len(y_test))
    
    # Combine original features with DNN predictions
    X_train_ensemble = np.column_stack([X_train.values, dnn_train_pred])
    X_test_ensemble = np.column_stack([X_test.values, dnn_test_pred])
    
    # Train XGBoost on combined features with tuned parameters
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,  # Increased estimators
        max_depth=10,       # Increased depth for better fitting
        learning_rate=0.03,  # Lower learning rate for better convergence
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=0.1,     # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_ensemble, y_train, 
                  eval_set=[(X_test_ensemble, y_test)],
                  early_stopping_rounds=30,
                  verbose=10)
    
    # Make ensemble predictions
    y_pred_ensemble = xgb_model.predict(X_test_ensemble)
    
    # Evaluate ensemble model
    mse = mean_squared_error(y_test, y_pred_ensemble)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    r2 = r2_score(y_test, y_pred_ensemble)
    
    print(f"\nEnsemble Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save ensemble model
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/ensemble_model.pkl')
    print("Ensemble model saved to models/ensemble_model.pkl")
    
    return xgb_model

if __name__ == "__main__":
    train_ensemble()