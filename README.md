# 🔋 EcoGrid: Microgrid Optimizer Prototype

EcoGrid is an AI-powered microgrid management system designed to optimize community energy self-sufficiency. By utilizing machine learning predictive modeling, the system dynamically forecasts local solar energy generation alongside smart-community consumption trends, coordinating a localized Battery Energy Storage System (BESS) to minimize main-grid dependency and maximize carbon offsets.

---

## 🚀 Features
* **Multi-Agent Simulation Pipeline:** Models a localized grid environment consisting of clean energy generation assets, community consumption loads, and a virtual battery storage unit.
* **Dual-Engine Predictive Modeling:** Leverages optimized Random Forest Regressors to dynamically forecast 24-hour supply generation efficiency and load metrics simultaneously.
* **Interactive Optimization Sandbox:** Built an intuitive user dashboard displaying real-time microgrid metrics, toggleable system capacities, dynamic energy flow visualization graphs, and translated language localized variations.

---

## 🛠️ System Architecture & Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Regression)
* **Data Pipelines:** Pandas, NumPy, Joblib
* **User Interface:** Streamlit Framework

---

## 📦 Project Structure & Workflow
1. `dataset_generator.py`: Generates 30-day hourly synthetic weather irradiance data and multi-peak community load curves.
2. `ai_engine.py`: Preprocesses telemetry, splits training datasets, fits the machine learning regressors, and exports optimized serialized model binaries (`.pkl`).
3. `app_runner.py`: Executes the core algorithmic rule-engine tracking logic, calculates real-time metrics (carbon avoided, grid fallback, energy waste), and serves the interactive Streamlit dashboard.

---

## ⚙️ How to Run Locally

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib streamlit