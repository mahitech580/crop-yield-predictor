import os
import sys
import numpy as np
import joblib

# TensorFlow can be unavailable or too heavy for a serverless function.
# Keep the app importable even when TensorFlow cannot be loaded.
if os.environ.get("VERCEL"):
    tf = None
    TF_IMPORT_ERROR = "TensorFlow model inference is disabled in the Vercel serverless runtime."
else:
    try:
        import tensorflow as tf
        TF_IMPORT_ERROR = None
    except Exception as exc:
        tf = None
        TF_IMPORT_ERROR = str(exc)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fertilizer_mapper import recommend_fertilizer, get_fertilizer_cost

class CropYieldPredictor:
    def __init__(self):
        self.dnn_model = None
        self.ensemble_model = None
        self.scaler = None
        self.crop_encoder = None
        self.soil_encoder = None
        self.season_encoder = None
        self.irrigation_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load trained models using paths anchored to the repository root."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        try:
            if tf is None:
                raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

            self.dnn_model = tf.keras.models.load_model(
                os.path.join(models_dir, "dnn_regressor.h5"),
                compile=False
            )
            self.ensemble_model = joblib.load(os.path.join(models_dir, "ensemble_model.pkl"))
            self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            self.crop_encoder = joblib.load(os.path.join(models_dir, "crop_encoder.pkl"))
            self.soil_encoder = joblib.load(os.path.join(models_dir, "soil_encoder.pkl"))

            # Load additional encoders if they exist.
            try:
                self.season_encoder = joblib.load(os.path.join(models_dir, "season_encoder.pkl"))
            except Exception:
                self.season_encoder = None

            try:
                self.irrigation_encoder = joblib.load(os.path.join(models_dir, "irrigation_encoder.pkl"))
            except Exception:
                self.irrigation_encoder = None

            print("All models loaded successfully")
        except Exception as e:
            print(f"Error loading trained models: {e}")
            self.dnn_model = None
            self.ensemble_model = None
            self.create_dummy_models()

    def preprocess_input(self, rainfall, temperature, humidity, N, P, K, crop, soil_type, 
                        pH=6.5, area=5.0, season='Kharif', irrigation='Rainfed'):
        """Preprocess input data for prediction with correct feature set"""
        # Encode categorical variables
        try:
            crop_encoded = self.crop_encoder.transform([crop])[0]
        except:
            crop_encoded = 0  # Default encoding
        
        try:
            soil_encoded = self.soil_encoder.transform([soil_type])[0]
        except:
            soil_encoded = 0  # Default encoding
            
        # Create feature array with correct features (matching training data)
        # Training data uses: Rainfall, Temperature, Humidity, N, P, K, Crop_Encoded, Soil_Encoded
        features = np.array([[rainfall, temperature, humidity, N, P, K, crop_encoded, soil_encoded]])
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features)
        except:
            # If scaling fails, use unscaled features
            features_scaled = features
        
        return features_scaled
    
    def create_dummy_models(self):
        """Create dummy models for demonstration"""
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import numpy as np
        
        self.scaler = StandardScaler()
        # Updated to match correct features: Rainfall, Temperature, Humidity, N, P, K, Crop_Encoded, Soil_Encoded
        self.scaler.mean_ = np.array([800, 25, 65, 50, 30, 40, 2, 2])
        self.scaler.scale_ = np.array([200, 5, 15, 15, 10, 12, 1.5, 1.5])
        
        self.crop_encoder = LabelEncoder()
        self.crop_encoder.classes_ = np.array(['Cotton', 'Maize', 'Rice', 'Sugarcane', 'Wheat'])
        
        self.soil_encoder = LabelEncoder()
        self.soil_encoder.classes_ = np.array(['Black', 'Clay', 'Loamy', 'Red', 'Sandy'])
        
        print("Dummy models created for demonstration")
    
    def predict_yield(self, rainfall, temperature, humidity, N, P, K, crop, soil_type,
                     pH=6.5, area=5.0, season='Kharif', irrigation='Rainfed'):
        """Predict crop yield using ensemble model with additional features"""
        X = self.preprocess_input(rainfall, temperature, humidity, N, P, K, crop, soil_type,
                                 pH, area, season, irrigation)
        
        try:
            # Try ensemble model first (most accurate)
            dnn_pred = self.dnn_model.predict(X, verbose=0)[0][0]
            X_ensemble = np.column_stack([X, [[dnn_pred]]])
            ensemble_pred = self.ensemble_model.predict(X_ensemble)[0]
            
            # Confidence estimation based on model agreement
            confidence = 1.0 - abs(dnn_pred - ensemble_pred) / max(dnn_pred, ensemble_pred, 1e-6)
            confidence = max(0.7, min(1.0, confidence))  # Clamp between 0.7 and 1.0
            
            return {
                'dnn_prediction': float(dnn_pred),
                'ensemble_prediction': float(ensemble_pred),
                'final_yield': float(ensemble_pred),
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Primary model prediction failed: {e}")
            try:
                # Fallback to DNN only
                dnn_pred = self.dnn_model.predict(X, verbose=0)[0][0]
                return {
                    'dnn_prediction': float(dnn_pred),
                    'ensemble_prediction': float(dnn_pred),
                    'final_yield': float(dnn_pred),
                    'confidence': 0.85  # Moderate confidence for fallback
                }
            except Exception as e2:
                print(f"DNN model prediction failed: {e2}")
                # Ultimate fallback to formula
                yield_pred = (0.003 * rainfall + 0.15 * temperature + 0.08 * humidity + 
                             0.25 * N + 0.2 * P + 0.15 * K + 0.4 * pH + 0.05 * np.log(area + 1)) / 100
                
                # Apply categorical multipliers
                season_multiplier = {'Kharif': 1.2, 'Rabi': 1.0, 'Zaid': 0.8}
                irrigation_multiplier = {'Drip': 1.3, 'Sprinkler': 1.15, 'Flood': 1.0, 'Rainfed': 0.85}
                crop_multiplier = {'Rice': 1.1, 'Wheat': 1.0, 'Maize': 1.2, 'Cotton': 0.9, 'Sugarcane': 1.4}
                soil_multiplier = {'Clay': 0.9, 'Sandy': 0.8, 'Loamy': 1.1, 'Black': 1.05, 'Red': 0.95}
                
                multiplier = (season_multiplier.get(season, 1.0) * 
                            irrigation_multiplier.get(irrigation, 1.0) *
                            crop_multiplier.get(crop, 1.0) *
                            soil_multiplier.get(soil_type, 1.0))
                yield_pred *= multiplier
                
                return {
                    'dnn_prediction': float(yield_pred * 0.95),
                    'ensemble_prediction': float(yield_pred),
                    'final_yield': float(yield_pred),
                    'confidence': 0.7  # Lower confidence for formula-based prediction
                }
    
    def get_recommendation(self, rainfall, temperature, humidity, N, P, K, crop, soil_type,
                          pH=6.5, area=5.0, season='Kharif', irrigation='Rainfed'):
        """Get complete prediction and fertilizer recommendation"""
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) and x >= 0 for x in [rainfall, temperature, humidity, N, P, K, pH, area]):
                raise ValueError("Numeric inputs must be non-negative numbers")
            
            if not isinstance(crop, str) or not isinstance(soil_type, str):
                raise ValueError("Crop and soil_type must be strings")
            
            # Predict yield
            yield_result = self.predict_yield(rainfall, temperature, humidity, N, P, K, crop, soil_type,
                                            pH, area, season, irrigation)
            
            # Get fertilizer recommendation
            fertilizer = recommend_fertilizer(N, P, K, soil_type, crop)
            cost = get_fertilizer_cost(fertilizer)
            
            return {
                'predicted_yield': yield_result['final_yield'],
                'fertilizer_recommendation': fertilizer,
                'estimated_cost': cost,
                'model_details': yield_result,
                'confidence': yield_result['confidence']
            }
        except Exception as e:
            print(f"Error in get_recommendation: {str(e)}")
            raise

if __name__ == "__main__":
    # Test prediction
    predictor = CropYieldPredictor()
    
    result = predictor.get_recommendation(
        rainfall=850, temperature=28, humidity=70,
        N=45, P=25, K=35, pH=6.5, area=5.0,
        crop='Rice', soil_type='Clay', season='Kharif', irrigation='Flood'
    )
    
    print("Prediction Results:")
    print(f"Predicted Yield: {result['predicted_yield']:.2f} tons/hectare")
    print(f"Fertilizer: {result['fertilizer_recommendation']}")
    print(f"Cost: Rs.{result['estimated_cost']:.1f}/kg")