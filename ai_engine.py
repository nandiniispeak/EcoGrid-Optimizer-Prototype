import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🔄 Loading grid data stream...")
data = pd.read_csv("grid_data.csv")

X = data[['Hour', 'Solar_Irradiance']]
y_generation = data['Solar_Irradiance'] * 0.18  # 18% solar conversion efficiency
y_load = data['Community_Demand']

print("🧠 Training Solar Generation Predictor Model...")
X_train, X_test, y_g_train, y_g_test = train_test_split(X, y_generation, test_size=0.2, random_state=42)
gen_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
gen_predictor.fit(X_train, y_g_train)

print("🧠 Training Smart Community Load Demand Model...")
X_train, X_test, y_l_train, y_l_test = train_test_split(X, y_load, test_size=0.2, random_state=42)
load_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
load_predictor.fit(X_train, y_l_train)

# Save our trained weights files
joblib.dump(gen_predictor, "generation_model.pkl")
joblib.dump(load_predictor, "demand_model.pkl")
print("✅ AI Engine weights compiled and exported successfully!")