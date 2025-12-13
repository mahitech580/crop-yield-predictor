import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def reshape_for_lstm(X, timesteps=3):
    """Reshape data for LSTM input"""
    samples = X.shape[0] - timesteps + 1
    features = X.shape[1]
    
    X_reshaped = np.zeros((samples, timesteps, features))
    
    for i in range(samples):
        X_reshaped[i] = X[i:i+timesteps]
    
    return X_reshaped

def build_lstm_model(timesteps, features):
    """Build LSTM regression model"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

def train_lstm():
    """Train LSTM model for temporal patterns"""
    # Load processed data
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train_data.drop('Yield', axis=1).values
    y_train = train_data['Yield'].values
    X_test = test_data.drop('Yield', axis=1).values
    y_test = test_data['Yield'].values
    
    # Reshape for LSTM
    timesteps = 5  # Increased timesteps for better temporal patterns
    X_train_lstm = reshape_for_lstm(X_train, timesteps)
    X_test_lstm = reshape_for_lstm(X_test, timesteps)
    
    # Adjust target arrays
    y_train_lstm = y_train[timesteps-1:]
    y_test_lstm = y_test[timesteps-1:]
    
    # Build and train model
    model = build_lstm_model(timesteps, X_train.shape[1])
    
    # Add callbacks for better training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
        verbose=1
    )
    
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,  # Increased epochs
        batch_size=16,  # Smaller batch size
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test_lstm)
    
    mse = mean_squared_error(y_test_lstm, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_lstm, y_pred)
    r2 = r2_score(y_test_lstm, y_pred)
    
    print(f"\nLSTM Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model.h5')
    print("LSTM model saved to models/lstm_model.h5")
    
    return model, history

if __name__ == "__main__":
    train_lstm()