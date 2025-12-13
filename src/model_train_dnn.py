import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def build_dnn_model(input_dim):
    """Build Deep Neural Network Regressor with improved architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

def train_dnn():
    """Train DNN Regressor model with enhanced training"""
    # Load processed data
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train_data.drop('Yield', axis=1)
    y_train = train_data['Yield']
    X_test = test_data.drop('Yield', axis=1)
    y_test = test_data['Yield']
    
    # Build and train model
    model = build_dnn_model(X_train.shape[1])
    
    # Add callbacks for better training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
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
    
    # Add model checkpoint callback
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/dnn_regressor_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=300,  # Increased epochs for better training
        batch_size=16,  # Smaller batch size for better generalization
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Load best model
    try:
        model = tf.keras.models.load_model('models/dnn_regressor_best.h5')
        print("Loaded best model from checkpoint")
    except:
        print("Using final model weights")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nDNN Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/dnn_regressor.h5')
    print("DNN model saved to models/dnn_regressor.h5")
    
    return model, history

if __name__ == "__main__":
    train_dnn()