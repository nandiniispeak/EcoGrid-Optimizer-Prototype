import streamlit as st
import pandas as pd
import joblib

class VirtualBatteryUnit:
    def __init__(self, limit, charge):
        self.capacity = limit
        self.soc = charge
    def charge(self, amt):
        loss = 0
        if self.soc + amt <= self.capacity: self.soc += amt
        else:
            loss = amt - (self.capacity - self.soc)
            self.soc = self.capacity
        return loss
    def discharge(self, amt):
        taken = 0
        if self.soc >= amt:
            self.soc -= amt
            taken = amt
        else:
            taken = self.soc
            self.soc = 0
        return taken

st.set_page_config(page_title="EcoGrid Prototype Launcher", layout="wide")
st.title("🔋 EcoGrid: Multi-Agent Microgrid Optimizer Prototype")

language = st.sidebar.selectbox("Interface Language / भाषा चुनिए", ["English", "Hindi"])
cap_slider = st.sidebar.slider("Battery Storage Capacity (kWh)" if language == "English" else "बैटरी स्टोरेज क्षमता (kWh)", 100, 1000, 400)

df = pd.read_csv("grid_data.csv").head(24)
gen_m = joblib.load("generation_model.pkl")
dem_m = joblib.load("demand_model.pkl")

bess = VirtualBatteryUnit(limit=cap_slider, charge=150)
logs = []

for idx, row in df.iterrows():
    h = int(row['Hour'])
    p_gen = gen_m.predict([[h, row['Solar_Irradiance']]])[0]
    p_dem = dem_m.predict([[h, row['Community_Demand']]])[0]
    diff = p_gen - p_dem
    wasted, fallback = 0, 0
    if diff > 0: wasted = bess.charge(diff)
    else: fallback = abs(diff) - bess.discharge(abs(diff))
    logs.append({"Hour": h, "Supply_kWh": p_gen, "Demand_kWh": p_dem, "Battery_Storage_kWh": bess.soc, "Wasted_kWh": wasted, "Grid_Fallback_kWh": fallback})

res_df = pd.DataFrame(logs)
co2_offset = (res_df['Supply_kWh'].sum() - res_df['Wasted_kWh'].sum()) * 0.82

c1, c2, c3 = st.columns(3)
c1.metric("Carbon Avoided" if language == "English" else "कार्बन उत्सर्जन में कमी", f"{co2_offset:.2f} kg CO2")
c2.metric("Wasted Energy" if language == "English" else "बर्बाद सौर ऊर्जा", f"{res_df['Wasted_kWh'].sum():.2f} kWh")
c3.metric("Grid Fallback" if language == "English" else "मुख्य ग्रिड निर्भरता", f"{res_df['Grid_Fallback_kWh'].sum():.2f} kWh")

st.markdown("### Dynamic Real-Time Energy Flow Graph")
st.line_chart(res_df.set_index("Hour")[["Supply_kWh", "Demand_kWh", "Battery_Storage_kWh"]])