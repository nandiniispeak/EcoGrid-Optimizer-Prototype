import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Replace 'microgrid_data.csv' with your actual generated filename if different
# 3. TRAIN/TEST SPLIT (80% Training, 20% Testing for validation)
# 1. LOAD DATA
try:
    df = pd.read_csv('grid_data.csv')
except FileNotFoundError:
    print("Error: grid_data.csv not found. Run dataset_generator.py first!")
    exit()

# 2. CORRECTED FEATURE ENGINEERING (Matching your exact columns)
# Input features: Hour of the day and Solar Irradiance
X = df[['Hour', 'Solar_Irradiance']]  

# Target variables we want to predict
y_solar = df['Solar_Irradiance']     # Predicting solar profiles
y_load = df['Community_Demand']      # Predicting demand load

X_train, X_test, y_solar_train, y_solar_test = train_test_split(X, y_solar, test_size=0.2, random_state=42)
_, _, y_load_train, y_load_test = train_test_split(X, y_load, test_size=0.2, random_state=42)

# 4. INITIALIZE AND TRAIN THE PARALLEL REGRESSORS
solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
load_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training Solar Supply Predictor...")
solar_model.fit(X_train, y_solar_train)

print("Training Load Demand Predictor...")
load_model.fit(X_train, y_load_train)

# 5. CALCULATE MEAN ABSOLUTE ERROR (MAE)
solar_preds = solar_model.predict(X_test)
load_preds = load_model.predict(X_test)

solar_mae = mean_absolute_error(y_solar_test, solar_preds)
load_mae = mean_absolute_error(y_load_test, load_preds)

print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Solar Generation Model MAE: {solar_mae:.2f} kW")
print(f"Community Load Model MAE:   {load_mae:.2f} kW\n")

# 6. SERIALIZE MODEL BINARIES & METRICS
# Save models
joblib.dump(solar_model, 'solar_regressor.pkl')
joblib.dump(load_model, 'load_regressor.pkl')

# Save the calculated metrics in a separate file for app_runner to read
metrics = {
    'solar_mae': solar_mae,
    'load_mae': load_mae
}
joblib.dump(metrics, 'model_metrics.pkl')
print("Successfully saved models and validation metrics to disk.")