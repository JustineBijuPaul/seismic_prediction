import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import joblib
import os

def create_initial_models():
    """Create and save initial models with dummy data for testing"""
    print("Creating initial models...")
    
    # Create dummy data
    n_samples = 100
    X_earthquake = np.random.randn(n_samples, 17)  # 17 features as per extract_features()
    y_earthquake = np.random.randint(0, 2, n_samples)
    
    X_time = np.random.randn(n_samples, 19)  # 19 features as per extract_temporal_features()
    y_time = np.random.exponential(24, n_samples)  # Hours prediction
    
    X_precursor = np.random.randn(n_samples, 19)  # Same features as temporal
    y_precursor = np.random.randint(0, 2, n_samples)
    
    # Train and save earthquake detection model
    eq_scaler = StandardScaler()
    X_earthquake_scaled = eq_scaler.fit_transform(X_earthquake)
    eq_model = LogisticRegression(random_state=42)
    eq_model.fit(X_earthquake_scaled, y_earthquake)
    
    # Train and save time prediction model
    time_scaler = StandardScaler()
    X_time_scaled = time_scaler.fit_transform(X_time)
    time_model = RandomForestRegressor(random_state=42)
    time_model.fit(X_time_scaled, y_time)
    
    # Train and save precursor detection model
    precursor_scaler = StandardScaler()
    X_precursor_scaled = precursor_scaler.fit_transform(X_precursor)
    precursor_model = GradientBoostingClassifier(random_state=42)
    precursor_model.fit(X_precursor_scaled, y_precursor)
    
    # Save all models and scalers
    model_files = {
        'earthquake_model.joblib': eq_model,
        'earthquake_scaler.joblib': eq_scaler,
        'time_prediction_model.joblib': time_model,
        'time_prediction_scaler.joblib': time_scaler,
        'precursor_detection_model.joblib': precursor_model,
        'precursor_detection_scaler.joblib': precursor_scaler
    }
    
    for filename, model in model_files.items():
        joblib.dump(model, filename)
        print(f"Saved {filename}")

if __name__ == '__main__':
    create_initial_models()
    print("\nAll initial models have been created and saved.")
    print("You can now run app.py")
