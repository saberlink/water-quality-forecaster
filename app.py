import os
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import reduce
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- 0. SYSTEM CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Smart Water AI: Forecast & Simulate", layout="wide", page_icon="💧")

# --- 1. GLOBAL CONSTANTS ---
SEQUENCE_LENGTH = 30
TARGET_COLS = ['Turbidity', 'EC']

# - Exact Defaults
DEFAULTS = {
    'pH': 7.0, 'Temperature': 20.0, 'Rainfall': 0.0, 
    'EC': 0.0, 'Turbidity': 0.0, 
    'Watercourse_Level': 0.0, 'Watercourse_Discharge': 0.0
}

# - Exact Standards
WQI_STANDARDS = {
    "Drinking Water": {
        "pH":           {"Sn_max": 8.5, "Sn_min": 6.5, "Vid": 7.0, "type": "range"},
        "TDS":          {"Sn": 500,  "Vid": 0,   "type": "normal"},
        "Turbidity":    {"Sn": 5,    "Vid": 0,   "type": "normal"},
        "DO":           {"Sn": 6,    "Vid": 14.6,"type": "inverse"},
        "BOD":          {"Sn": 5,    "Vid": 0,   "type": "normal"},
        "TotalHardness":{"Sn": 300,  "Vid": 0,   "type": "normal"},
        "Chlorides":    {"Sn": 250,  "Vid": 0,   "type": "normal"},
        "Nitrates":     {"Sn": 45,   "Vid": 0,   "type": "normal"},
        "Sulphates":    {"Sn": 250,  "Vid": 0,   "type": "normal"},
        "EC":           {"Sn": 300,  "Vid": 0,   "type": "normal"}
    },
    "Agriculture": {
        "pH":           {"Sn_max": 8.4, "Sn_min": 6.5, "Vid": 7.0, "type": "range"},
        "TDS":          {"Sn": 2000, "Vid": 0,   "type": "normal"},
        "EC":           {"Sn": 3,    "Vid": 0,   "type": "normal"},
        "SAR":          {"Sn": 18,   "Vid": 0,   "type": "normal"},
        "Nitrates":     {"Sn": 30,   "Vid": 0,   "type": "normal"},
        "Chlorides":    {"Sn": 350,  "Vid": 0,   "type": "normal"},
        "Turbidity":    {"Sn": 50,   "Vid": 0,   "type": "normal"}
    },
    "Mining": {
        "pH":           {"Sn_max": 9.0, "Sn_min": 6.0, "Vid": 7.0, "type": "range"},
        "TDS":          {"Sn": 3000, "Vid": 0,   "type": "normal"},
        "TSS":          {"Sn": 50,   "Vid": 0,   "type": "normal"},
        "Turbidity":    {"Sn": 50,   "Vid": 0,   "type": "normal"},
        "Iron":         {"Sn": 2.0,  "Vid": 0,   "type": "normal"},
        "Sulphates":    {"Sn": 1000, "Vid": 0,   "type": "normal"},
        "EC":           {"Sn": 200,  "Vid": 0,   "type": "normal"}
    }
}

# --- 2. CORE FUNCTIONS ---

def calculate_wqi_precise(row, use_case="Drinking Water"):
    """
    - Exact Logic
    Calculates WQI for a single row using precise Harmonic Mean logic.
    """
    standards = WQI_STANDARDS.get(use_case)
    if not standards: return None
    available_params = {}
    for param, stats in standards.items():
        matching_col = next((col for col in row.index if col.lower() == param.lower()), None)
        if matching_col and pd.notna(row[matching_col]):
            available_params[param] = {'stats': stats, 'value': float(row[matching_col])}
    if not available_params: return 0
    sum_inverse_sn = 0
    for param, data in available_params.items():
        stats = data['stats']
        sn_effective = stats['Sn_max'] - stats['Vid'] if stats['type'] == 'range' else \
                       stats['Vid'] - stats['Sn'] if stats['type'] == 'inverse' else stats['Sn']
        if sn_effective != 0: sum_inverse_sn += (1 / sn_effective)
    K = 1 / sum_inverse_sn if sum_inverse_sn != 0 else 1
    numerator, denominator = 0, 0
    for param, data in available_params.items():
        stats = data['stats']
        vn = data['value']
        sn_effective = stats['Sn_max'] - stats['Vid'] if stats['type'] == 'range' else \
                       stats['Vid'] - stats['Sn'] if stats['type'] == 'inverse' else stats['Sn']
        wn = K / sn_effective if (sn_effective != 0 and K != 1) else 0.1
        qn = 0
        if stats['type'] == 'normal': qn = 100 * ((vn - stats['Vid']) / (stats['Sn'] - stats['Vid']))
        elif stats['type'] == 'range':
            max_deviation = max(stats['Sn_max'] - stats['Vid'], stats['Vid'] - stats['Sn_min'])
            qn = 100 * (abs(vn - stats['Vid']) / max_deviation)
        elif stats['type'] == 'inverse': qn = 100 * ((stats['Vid'] - vn) / (stats['Vid'] - stats['Sn']))
        numerator += (wn * max(0, qn))
        denominator += wn
    return numerator / denominator if denominator != 0 else 0

def process_uploaded_files(uploaded_files):
    """
    - Exact Logic
    Phase 1: Merge raw files and fix timestamps.
    """
    # STRICTLY MATCHING APP_9.PY FILENAMES ONLY
    filename_map = {
        "EC": ["ec", "conductivity"], "Turbidity": ["turb", "turbidity"],
        "Rainfall": ["rain", "rainfall", "precip"], "pH": ["ph"],
        "Temperature": ["temp", "temperature"], 
        "Watercourse_Level": ["level", "height", "stage"],
        "Watercourse_Discharge": ["discharge", "flow"]
    }
    found_files = {}
    
    # Identify Files based on filename keywords
    for file in uploaded_files:
        name_lower = file.name.lower()
        for system_name, keywords in filename_map.items():
            if any(keyword in name_lower for keyword in keywords):
                found_files[system_name] = file
                break
    
    data_frames = []

    # Read Files
    for param, file in found_files.items():
        try:
            # Skip metadata lines (#)
            df = pd.read_csv(file, comment='#', header=None)
            df = df.iloc[:, [0, 1]] 
            df.columns = ['Timestamp', param]
            
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            df[param] = pd.to_numeric(df[param], errors='coerce')
            
            # Set index for merging
            df.set_index('Timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            data_frames.append(df)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            return None

    if not data_frames: return None
    
    # Merge and Resample to 15min
    merged_df = pd.concat(data_frames, axis=1)
    merged_df = merged_df.resample('15min').mean()
    
    merged_df.reset_index(inplace=True)
    merged_df = merged_df.sort_values(by='Timestamp')
    return merged_df

def clean_and_fill_data(df):
    """
    - Exact Logic
    Phase 2: Interpolate and Clean.
    """
    # Ensure all expected columns exist
    for col, default_val in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val 
        else:
            if col == 'Rainfall':
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].fillna(default_val)
                
    df.dropna(subset=['Timestamp'], inplace=True)
    return df

def engineer_features(df):
    """
    - Exact Logic
    Phase 3: Add AI Memory Features (Lags).
    """
    df_eng = df.copy()
    
    # Time
    df_eng['Month'] = df_eng['Timestamp'].dt.month
    df_eng['DayOfWeek'] = df_eng['Timestamp'].dt.dayofweek
    df_eng['DayOfYear'] = df_eng['Timestamp'].dt.dayofyear
    
    # Lags
    cols_to_lag = ['EC', 'Turbidity', 'pH', 'Temperature', 'Watercourse_Level', 'Watercourse_Discharge']
    for col in cols_to_lag:
        if col in df_eng.columns:
            df_eng[f'{col}_lag1'] = df_eng[col].shift(1)
        else:
            df_eng[f'{col}_lag1'] = 0

    # Rolling
    if 'Rainfall' in df_eng.columns:
        df_eng['Rainfall_rolling_mean_7d'] = df_eng['Rainfall'].rolling(window=7, min_periods=1).mean()
    if 'Turbidity' in df_eng.columns:
        df_eng['Turbidity_rolling_mean_7d'] = df_eng['Turbidity'].rolling(window=7, min_periods=1).mean()
    if 'Watercourse_Discharge' in df_eng.columns:
        df_eng['Discharge_rolling_mean_7d'] = df_eng['Watercourse_Discharge'].rolling(window=7, min_periods=1).mean()

    return df_eng

def run_prediction_loop(model, clean_df, eng_df, days, scaler_X, scaler_y, feature_cols, use_case, manual_overrides=None):
    """
    Runs the prediction loop (Used for both Baseline and Simulator).
    """
    future_predictions = []
    running_history = clean_df.copy()
    current_eng_buffer = eng_df.copy()
    
    for i in range(days):
        last_30 = current_eng_buffer.iloc[-SEQUENCE_LENGTH:]
        X_in = scaler_X.transform(last_30[feature_cols].values).reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        
        pred = scaler_y.inverse_transform(model.predict(X_in, verbose=0))[0]
        p_turb, p_ec = max(0, pred[0]), max(0, pred[1])
        
        next_date = running_history['Timestamp'].max() + pd.Timedelta(days=1)
        new_row = {'Timestamp': next_date}
        last_vals = running_history.iloc[-1]
        
        for col in running_history.columns:
            if col == 'Timestamp': continue
            
            # --- OVERRIDE LOGIC ---
            if manual_overrides and col in manual_overrides:
                 new_row[col] = manual_overrides[col]
            # ----------------------
            
            elif col == 'Turbidity': new_row[col] = p_turb
            elif col == 'EC': new_row[col] = p_ec
            elif 'Rain' in col: new_row[col] = 0
            else: new_row[col] = last_vals[col]
        
        new_row['WQI'] = calculate_wqi_precise(pd.Series(new_row), use_case)
        future_predictions.append(new_row)
        
        running_history = pd.concat([running_history, pd.DataFrame([new_row])], ignore_index=True)
        current_eng_buffer = engineer_features(running_history.copy())

    return pd.DataFrame(future_predictions)

# --- 3. MAIN APPLICATION UI ---

st.title("💧 Smart Water AI: Forecast & Simulate")

with st.sidebar:
    st.header("1. Configuration")
    use_case = st.selectbox("WQI Standard", ["Drinking Water", "Agriculture", "Mining"])
    days_to_predict = st.slider("Forecast Horizon (Days)", 1, 14, 7)
    
    st.divider()
    st.header("2. Upload Data")
    uploaded_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, type=['csv'])

if 'clean_history' not in st.session_state: st.session_state.clean_history = None
if 'baseline_forecast' not in st.session_state: st.session_state.baseline_forecast = None
if 'model_assets' not in st.session_state: st.session_state.model_assets = None

# ==========================================
# STEP 1: LOAD & VISUALIZE (Strict Match to App 9)
# ==========================================
st.header("Step 1: Process & Verify Data")
if st.button("🚀 Process Data"):
    if uploaded_files:
        with st.spinner('Processing...'):
            raw = process_uploaded_files(uploaded_files)
            if raw is not None:
                st.session_state.clean_history = clean_and_fill_data(raw)
                st.success(f"Loaded {len(st.session_state.clean_history)} records.")
                
                # Visualization (App 9 Style)
                df = st.session_state.clean_history
                numeric_cols = [c for c in df.columns if c != 'Timestamp']
                if numeric_cols:
                    for col in numeric_cols:
                        fig = go.Figure()
                        if 'Rain' in col:
                            fig.add_trace(go.Bar(x=df['Timestamp'], y=df[col], name=col, marker_color='blue'))
                        else:
                            fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[col], mode='lines', name=col))
                        fig.update_layout(title=f"{col} History", height=300, margin=dict(t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# STEP 2: BASELINE FORECAST (Strict Match to App 9)
# ==========================================
st.divider()
st.header("Step 2: Run Baseline AI Prediction")

if st.button("🧠 Run Forecast"):
    if st.session_state.clean_history is not None:
        with st.spinner('Forecasting...'):
            df = st.session_state.clean_history
            eng = engineer_features(df.copy())
            
            # Padding
            if len(eng) < SEQUENCE_LENGTH:
                st.warning(f"Data too short ({len(eng)} rows). Auto-padding to fit AI model.")
                pad = pd.concat([eng.iloc[[0]]] * (SEQUENCE_LENGTH - len(eng)), ignore_index=True)
                eng = pd.concat([pad, eng], ignore_index=True)
            
            try:
                model = load_model('multi_output_model.keras')
                # Identify features
                f_cols = [c for c in eng.columns if c not in TARGET_COLS and c != 'Timestamp']
                if len(f_cols) > 17: f_cols = f_cols[:17] # Match App 9 safety clip
                
                scaler_X = MinMaxScaler().fit(eng[f_cols].values)
                scaler_y = MinMaxScaler().fit(df[TARGET_COLS].values)
                
                st.session_state.model_assets = (model, scaler_X, scaler_y, f_cols)
                
                baseline = run_prediction_loop(model, df, eng, days_to_predict, scaler_X, scaler_y, f_cols, use_case, None)
                st.session_state.baseline_forecast = baseline
                
                # Detailed Results
                st.subheader(f"Forecast Results ({days_to_predict} Days)")
                cols = st.columns(5)
                cols[0].metric("Avg WQI", f"{baseline['WQI'].mean():.1f}")
                cols[1].metric("Max Turbidity", f"{baseline['Turbidity'].max():.1f} NTU")
                cols[2].metric("Avg Turbidity", f"{baseline['Turbidity'].mean():.1f} NTU")
                cols[3].metric("Max EC", f"{baseline['EC'].max():.1f} uS/cm")
                cols[4].metric("Avg EC", f"{baseline['EC'].mean():.1f} uS/cm")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=baseline['Timestamp'], y=baseline['WQI'], mode='lines+markers', name='WQI'))
                fig.add_hrect(y0=0, y1=25, fillcolor="green", opacity=0.1, annotation_text="Excellent")
                fig.add_hrect(y0=50, y1=100, fillcolor="red", opacity=0.1, annotation_text="Poor")
                fig.update_layout(title="Predicted Water Quality Index (Overall Risk)", height=350)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(x=baseline['Timestamp'], y=baseline['Turbidity'], mode='lines', name='Turbidity', line=dict(color='orange')))
                    fig_t.update_layout(title="Predicted Turbidity", height=300)
                    st.plotly_chart(fig_t, use_container_width=True)
                with c2:
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Scatter(x=baseline['Timestamp'], y=baseline['EC'], mode='lines', name='EC', line=dict(color='purple')))
                    fig_e.update_layout(title="Predicted Conductivity (EC)", height=300)
                    st.plotly_chart(fig_e, use_container_width=True)

                csv = baseline.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", csv, "wqi_forecast.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Model Error: {e}")

# ==========================================
# STEP 3: EXPLAIN (SHAP XAI)
# ==========================================
st.divider()
st.header("Step 3: Explain Model Logic (XAI)")

if st.button("🔎 Run SHAP Analysis"):
    if st.session_state.model_assets is None:
        st.error("Please run Step 2 first.")
    else:
        with st.spinner('Calculating Feature Importance...'):
            try:
                model, scaler_X, _, f_cols = st.session_state.model_assets
                df = st.session_state.clean_history
                eng_df = engineer_features(df.copy())
                
                X = eng_df[f_cols].values
                X_scaled = scaler_X.transform(X)
                X_sequences = np.array([X_scaled[i:i + SEQUENCE_LENGTH] for i in range(len(X_scaled) - SEQUENCE_LENGTH)])
                X_train_2d = X_sequences.reshape(X_sequences.shape[0], -1)
                
                def predict_fn_2d(X_2d):
                    X_3d = X_2d.reshape(X_2d.shape[0], SEQUENCE_LENGTH, len(f_cols))
                    return model.predict(X_3d)

                bg_indices = np.random.choice(X_train_2d.shape[0], min(50, X_train_2d.shape[0]), replace=False)
                background_data_2d = X_train_2d[bg_indices]
                explainer = shap.KernelExplainer(predict_fn_2d, background_data_2d)
                
                shap_values = explainer.shap_values(X_train_2d[-5:])
                
                c1, c2 = st.columns(2)
                targets = ['Turbidity', 'EC']
                
                for i, target in enumerate(targets):
                    vals = np.mean(np.abs(shap_values[i].reshape(-1, SEQUENCE_LENGTH, len(f_cols))), axis=(0, 1))
                    shap_df = pd.DataFrame({'feature': f_cols, 'importance': vals}).sort_values('importance')
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(shap_df['feature'], shap_df['importance'], color='dodgerblue')
                    ax.set_title(f'Importance for {target}')
                    ax.set_xlabel('Impact')
                    
                    if i == 0: c1.pyplot(fig)
                    else: c2.pyplot(fig)
                st.success("✅ SHAP Analysis Complete.")
                    
            except Exception as e:
                st.error(f"XAI Error: {e}")

# ==========================================
# STEP 4: INTERACTIVE SIMULATOR
# ==========================================
st.divider()
st.header("Step 4: Interactive 'What-If' Simulator")
st.markdown("Adjust sliders to see how sensor values impact the forecast.")

if st.session_state.baseline_forecast is not None:
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🛠️ Scenario Builder")
        
        df_ref = st.session_state.clean_history
        # Only expose columns that actually exist in the strict 7-param limit of App 9
        input_cols = [c for c in df_ref.columns if c not in ['Timestamp', 'Turbidity', 'EC', 'WQI']]
        
        manual_overrides = {}
        
        for col in input_cols:
            min_val = float(df_ref[col].min())
            mean_val = float(df_ref[col].mean())
            max_val = float(df_ref[col].max())
            slider_max = max(10.0, max_val * 2.0)
            
            val = st.slider(f"{col}", 0.0, slider_max, mean_val, key=f"sim_{col}")
            manual_overrides[col] = val
        
        if st.button("▶️ Run Simulation"):
            with st.spinner("Simulating..."):
                model, sX, sy, fCols = st.session_state.model_assets
                clean_df = st.session_state.clean_history
                eng_df = engineer_features(clean_df.copy())
                
                sim_results = run_prediction_loop(model, clean_df, eng_df, days_to_predict, sX, sy, fCols, use_case, manual_overrides)
                st.session_state.sim_results = sim_results
                st.success("Simulation Updated!")

    with col2:
        if 'sim_results' in st.session_state:
            st.subheader("📊 Scenario Comparison")
            
            baseline = st.session_state.baseline_forecast
            simulated = st.session_state.sim_results
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=baseline['Timestamp'], y=baseline['Turbidity'], name='Baseline (Turb)', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=simulated['Timestamp'], y=simulated['Turbidity'], name='Simulated (Turb)', line=dict(color='orange', dash='dash', width=3)))
            fig.update_layout(title="Turbidity Impact", height=300, yaxis_title="NTU")
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=baseline['Timestamp'], y=baseline['EC'], name='Baseline (EC)', line=dict(color='gray')))
            fig2.add_trace(go.Scatter(x=simulated['Timestamp'], y=simulated['EC'], name='Simulated (EC)', line=dict(color='purple', dash='dash', width=3)))
            fig2.update_layout(title="EC Impact", height=300, yaxis_title="uS/cm")
            st.plotly_chart(fig2, use_container_width=True)

else:

    st.info("Please run Step 2 (Baseline Forecast) first.")
