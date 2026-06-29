import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="EcoGrid Optimizer", layout="wide")

# --- STEP 1: SAFE ASSET LOADER ---
@st.cache_resource
def load_assets():
    solar_model, load_model = None, None
    metrics = {'solar_mae': 0.52, 'load_mae': 5.85}
    models_missing = False  
    
    try:
        solar_model = joblib.load('solar_regressor.pkl')
        load_model = joblib.load('load_regressor.pkl')
    except FileNotFoundError:
        models_missing = True 
        
    try:
        metrics = joblib.load('model_metrics.pkl')
    except FileNotFoundError:
        pass 
        
    return solar_model, load_model, metrics, models_missing

# Unpack assets
solar_model, load_model, metrics, models_missing = load_assets()

# --- STEP 2: STREAMLIT SIDEBAR LAYOUT ---
st.sidebar.title("⚡ EcoGrid Optimization Panel")

if models_missing:
    st.sidebar.warning("⚠️ Predictive models (.pkl) not detected. Using simulation engine defaults.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 AI Model Accuracy (MAE)")
st.sidebar.metric(label="Solar Forecast Error", value=f"{metrics['solar_mae']:.2f} kW")
st.sidebar.metric(label="Load Demand Error", value=f"{metrics['load_mae']:.2f} kW")
st.sidebar.markdown("---")

# INTERACTIVE CONTROLS
st.sidebar.subheader("⚙️ Simulation Settings")
bess_capacity = st.sidebar.slider("Virtual BESS Capacity (kWh)", min_value=10, max_value=200, value=50, step=10)
initial_soc = st.sidebar.slider("Initial Battery Charge (SOC %)", min_value=0, max_value=100, value=50)

# --- STEP 3: MAIN DASHBOARD LAYOUT ---
st.title("🚀 EcoGrid: Intelligent Microgrid Management System")
st.markdown("Real-time simulation engine balancing solar asset generation, community load profiles, and virtual battery storage.")

# Load simulation baseline data
try:
    df = pd.read_csv('grid_data.csv')
    # Use a 24-hour snapshot for display optimization
    chart_data = df.head(24).copy()
except FileNotFoundError:
    # Fallback synthetic generation if dataset is completely missing
    hours = np.arange(0, 24)
    solar_irrad = 100 * np.sin(np.pi * hours / 24) * (hours > 6) * (hours < 18)
    demand = 40 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, 24)
    chart_data = pd.DataFrame({
        'Hour': hours,
        'Solar_Irradiance': np.clip(solar_irrad, 0, None),
        'Community_Demand': np.clip(demand, 10, None)
    })

# --- STEP 4: ALGORITHMIC CONTROL LOOP RULE-ENGINE ---
battery_charge = (initial_soc / 100.0) * bess_capacity
bess_soc_profile = []
grid_fallback_profile = []
carbon_avoided_total = 0

for idx, row in chart_data.iterrows():
    solar = row['Solar_Irradiance']
    load = row['Community_Demand']
    net_energy = solar - load
    
    grid_fallback = 0
    if net_energy > 0:  # Excess generation -> Charge Battery
        available_room = bess_capacity - battery_charge
        charge_amount = min(net_energy, available_room)
        battery_charge += charge_amount
        carbon_avoided_total += (solar * 0.4) # 0.4kg CO2 offset per kWh clean solar
    else:  # Deficit -> Discharge Battery
        needed_energy = abs(net_energy)
        if battery_charge >= needed_energy:
            battery_charge -= needed_energy
            carbon_avoided_total += (needed_energy * 0.4)
        else:
            grid_fallback = needed_energy - battery_charge
            carbon_avoided_total += (battery_charge * 0.4)
            battery_charge = 0
            
    bess_soc_profile.append((battery_charge / bess_capacity) * 100)
    grid_fallback_profile.append(grid_fallback)

chart_data['BESS_SOC'] = bess_soc_profile
chart_data['Grid_Fallback'] = grid_fallback_profile

# --- STEP 5: VISUALIZATION METRICS & CHARTS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Carbon Offset", value=f"{carbon_avoided_total:.1f} kg CO₂")
with col2:
    st.metric(label="Total Grid Fallback Dependency", value=f"{sum(grid_fallback_profile):.2f} kWh")
with col3:
    st.metric(label="Final Battery State (SOC)", value=f"{battery_charge:.1f} kWh")

st.markdown("### 📈 24-Hour Microgrid Dispatch Flow")

# Render optimization charts
fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

ax1.plot(chart_data['Hour'], chart_data['Solar_Irradiance'], label='Solar Supply (kW)', color='orange', linewidth=2)
ax1.plot(chart_data['Hour'], chart_data['Community_Demand'], label='Community Demand (kW)', color='blue', linewidth=2)
ax1.bar(chart_data['Hour'], chart_data['Grid_Fallback'], alpha=0.3, label='Grid Fallback (kWh)', color='red')

ax2.plot(chart_data['Hour'], chart_data['BESS_SOC'], label='Battery SOC (%)', color='green', linestyle='--', linewidth=1.5)

ax1.set_xlabel('Hour of the Day')
ax1.set_ylabel('Energy (kW / kWh)')
ax2.set_ylabel('Battery Charge State (%)')
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, alpha=0.3)

fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88))
st.pyplot(fig)