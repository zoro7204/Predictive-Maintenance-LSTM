# app_streamlit.py
"""
Predictive Maintenance — RUL Prediction Dashboard (Production-Ready)

HOW TO RUN:
    streamlit run app_streamlit.py

REQUIREMENTS:
    - Streamlit, pandas, numpy, matplotlib, tensorflow, joblib
    - Project files: case_study.py, evaluate_model.py, hybrid_model.py
    - Data files: test_FD001.txt, RUL_FD001.txt (optional)
    - Model files: hybrid_cnn_lstm_model.h5, scaler.pkl (optional)

FEATURES:
    - Single-engine RUL prediction via case_study.py subprocess
    - Full evaluation via evaluate_model.py subprocess
    - Model/scaler upload capability
    - Responsive card-based layout with modern styling
    - Real-time sensor visualization
    - Batch evaluation on test set
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from hybrid_model import create_hybrid_model
import subprocess
import re
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Predictive Maintenance — RUL Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    /* Modern font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    }
    
    /* Card-like panels */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Compact spacing */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Remove extra padding */
    .main .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-success {
        background: rgba(0, 200, 0, 0.2);
        color: #00ff00;
    }
    
    .status-warning {
        background: rgba(255, 165, 0, 0.2);
        color: #ffa500;
    }
    
    .status-error {
        background: rgba(255, 0, 0, 0.2);
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
SEQUENCE_LENGTH = 30
MODEL_PATH_DEFAULT = "hybrid_cnn_lstm_model.h5"
SCALER_PATH_DEFAULT = "scaler.pkl"
TEST_DATA_PATH = "test_FD001.txt"
RUL_PATH = "RUL_FD001.txt"
EVAL_PLOT = "rul_plot_final_evaluation.png"

# CMAPSS FD sensor column names (21 sensors)
sensor_cols = [f"sensor_measurement_{i+1}" for i in range(21)]

# ============================================================================
# CACHED RESOURCE LOADERS
# ============================================================================
@st.cache_resource
def load_scaler(path):
    """Load and cache the feature scaler"""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load scaler from {path}: {e}")
        return None

@st.cache_resource
def load_test_data(path):
    """Load and cache test dataset"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        col_names = ["engine_id", "cycle"] + [f"op_setting_{i+1}" for i in range(3)] + sensor_cols
        if df.shape[1] >= len(col_names):
            df = df.iloc[:, :len(col_names)]
            df.columns = col_names
            return df
        else:
            st.error("Test data doesn't have expected number of columns.")
            return None
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None

@st.cache_resource
def load_true_rul(path):
    """Load and cache true RUL values"""
    if not os.path.exists(path):
        return None
    return np.loadtxt(path)

@st.cache_resource
def build_and_load_model(model_path, input_shape=(SEQUENCE_LENGTH, len(sensor_cols))):
    """Build and load the trained model"""
    if not os.path.exists(model_path):
        return None
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception:
        try:
            num_stats = len(sensor_cols) * 4
            model = create_hybrid_model(input_shape, num_stats)
            model.load_weights(model_path)
            return model
        except Exception as e:
            st.error(f"Failed to load model from {model_path}: {e}")
            return None

# ============================================================================
# SUBPROCESS HELPERS
# ============================================================================
def predict_using_case_study(engine_id, cycle=None, python_exe="python"):
    """
    Call case_study.py as subprocess to get RUL prediction.
    Returns: (predicted_value, raw_output_text)
    """
    cmd = [python_exe, "case_study.py", "--engine_id", str(engine_id)]
    if cycle is not None:
        cmd += ["--cycle", str(cycle)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=120)
    except subprocess.TimeoutExpired:
        return None, "case_study.py timed out"
    
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    matches = re.findall(r"Predicted RUL[:\s]+([-+]?[0-9]*\.?[0-9]+)", out)
    if matches:
        try:
            val = float(matches[-1])
            return val, out
        except Exception:
            return None, out
    return None, out

def run_evaluate_model(timeout_seconds=600, python_exe="python"):
    """
    Run evaluate_model.py as subprocess.
    Returns: (success_bool, stdout_text)
    """
    cmd = [python_exe, "evaluate_model.py"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return False, "evaluate_model.py timed out."
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return True, out

def parse_metrics_from_output(output_text):
    """Parse MAE and RMSE from evaluate_model.py output"""
    mae_match = re.search(r"MAE[:\s]+([-+]?[0-9]*\.?[0-9]+)", output_text)
    rmse_match = re.search(r"RMSE[:\s]+([-+]?[0-9]*\.?[0-9]+)", output_text)
    mae = float(mae_match.group(1)) if mae_match else None
    rmse = float(rmse_match.group(1)) if rmse_match else None
    return mae, rmse

# ============================================================================
# SIDEBAR - FILE UPLOADS & SETTINGS
# ============================================================================
st.sidebar.markdown("### 📁 Model & Data Files")

model_upload = st.sidebar.file_uploader("Upload Model (.h5)", type=["h5"], help="Optional: upload custom trained model")
scaler_upload = st.sidebar.file_uploader("Upload Scaler (.pkl)", type=["pkl", "joblib"], help="Optional: upload feature scaler")

use_default = st.sidebar.checkbox("Use default project files", value=True, help="Load model/scaler/data from project directory")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This dashboard predicts Remaining Useful Life (RUL) for turbofan engines "
    "using NASA C-MAPSS FD001 dataset with a hybrid CNN-LSTM model."
)

# Handle uploaded files
model_path = MODEL_PATH_DEFAULT
scaler_path = SCALER_PATH_DEFAULT

if model_upload is not None:
    model_path = "uploaded_model.h5"
    with open(model_path, "wb") as f:
        f.write(model_upload.getbuffer())
    st.sidebar.success("✓ Model uploaded")

if scaler_upload is not None:
    scaler_path = "uploaded_scaler.pkl"
    with open(scaler_path, "wb") as f:
        f.write(scaler_upload.getbuffer())
    st.sidebar.success("✓ Scaler uploaded")

if not model_upload and os.path.exists(MODEL_PATH_DEFAULT):
    model_path = MODEL_PATH_DEFAULT

if not scaler_upload and os.path.exists(SCALER_PATH_DEFAULT):
    scaler_path = SCALER_PATH_DEFAULT

# ============================================================================
# LOAD RESOURCES
# ============================================================================
scaler = load_scaler(scaler_path) if os.path.exists(scaler_path) else None
df_test = load_test_data(TEST_DATA_PATH) if use_default else None
true_rul = load_true_rul(RUL_PATH) if os.path.exists(RUL_PATH) else None
model = build_and_load_model(model_path) if os.path.exists(model_path) else None

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🔧 Predictive Maintenance Dashboard</h1>
    <p class="subtitle">Remaining Useful Life (RUL) Prediction for Turbofan Engines</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# STATUS OVERVIEW CARD
# ============================================================================
st.markdown('<div class="info-card">', unsafe_allow_html=True)
col_status1, col_status2, col_status3, col_status4 = st.columns(4)

with col_status1:
    if model is not None:
        st.markdown('<span class="status-badge status-success">✓ Model Loaded</span>', unsafe_allow_html=True)
        model_name = getattr(model, "name", "Hybrid CNN-LSTM")
        st.caption(f"**{model_name}**")
    else:
        st.markdown('<span class="status-badge status-error">✗ No Model</span>', unsafe_allow_html=True)
        st.caption("Upload or check project folder")

with col_status2:
    if scaler is not None:
        st.markdown('<span class="status-badge status-success">✓ Scaler Loaded</span>', unsafe_allow_html=True)
        st.caption("Feature scaling ready")
    else:
        st.markdown('<span class="status-badge status-warning">⚠ No Scaler</span>', unsafe_allow_html=True)
        st.caption("Predictions may be inaccurate")

with col_status3:
    if df_test is not None:
        st.markdown('<span class="status-badge status-success">✓ Test Data</span>', unsafe_allow_html=True)
        st.caption(f"{len(df_test['engine_id'].unique())} engines")
    else:
        st.markdown('<span class="status-badge status-error">✗ No Test Data</span>', unsafe_allow_html=True)
        st.caption("test_FD001.txt not found")

with col_status4:
    if true_rul is not None:
        st.markdown('<span class="status-badge status-success">✓ True RUL</span>', unsafe_allow_html=True)
        st.caption(f"{len(true_rul)} ground truth values")
    else:
        st.markdown('<span class="status-badge status-warning">⚠ No True RUL</span>', unsafe_allow_html=True)
        st.caption("RUL_FD001.txt not found")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT - TWO COLUMN LAYOUT
# ============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

# ============================================================================
# LEFT COLUMN - SINGLE ENGINE PREDICTION
# ============================================================================
with col_left:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Single Engine Prediction")
    
    if df_test is None:
        st.warning("📊 Test dataset not available. Upload test_FD001.txt to enable predictions.")
        engine_id = None
        cycle = None
    else:
        engines = sorted(df_test["engine_id"].unique())
        
        col_eng, col_cyc = st.columns(2)
        with col_eng:
            engine_id = st.selectbox("Engine ID", engines, key="engine_select")
        
        with col_cyc:
            if engine_id is not None:
                max_cycle = int(df_test[df_test["engine_id"] == engine_id]["cycle"].max())
                cycle = st.slider("Cycle", min_value=SEQUENCE_LENGTH, max_value=max_cycle, 
                                value=max(SEQUENCE_LENGTH, max_cycle // 2), key="cycle_slider")
            else:
                cycle = SEQUENCE_LENGTH
        
        st.markdown("")  # Spacing
        
        if st.button("🚀 Predict RUL", use_container_width=True, type="primary"):
            if engine_id is None:
                st.error("⚠️ Please select an engine first.")
            else:
                with st.spinner("🔄 Running prediction via case_study.py..."):
                    pred_val, raw_output = predict_using_case_study(engine_id, cycle)
                
                if pred_val is None:
                    st.error("❌ Prediction failed. Check debug output below.")
                    with st.expander("🔍 Debug Output"):
                        st.code(raw_output[:4000], language="text")
                else:
                    st.success("✓ Prediction Complete")
                    
                    col_pred, col_true = st.columns(2)
                    with col_pred:
                        st.metric(label="Predicted RUL", value=f"{pred_val:.1f}", delta="cycles")
                    
                    with col_true:
                        if true_rul is not None and engine_id - 1 < len(true_rul):
                            true_val = int(true_rul[engine_id - 1])
                            error = pred_val - true_val
                            st.metric(label="True RUL", value=f"{true_val}", delta=f"{error:+.1f} error")
                        else:
                            st.metric(label="True RUL", value="N/A")
                    
                    with st.expander("📋 View Raw Output"):
                        st.code(raw_output, language="text")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SENSOR VISUALIZATION
    # ========================================================================
    if df_test is not None and engine_id is not None:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Sensor Readings (Last 30 Cycles)")
        
        engine_df_preview = df_test[df_test["engine_id"] == engine_id].sort_values("cycle")
        sample_window = engine_df_preview[engine_df_preview["cycle"] <= min(cycle, engine_df_preview["cycle"].max())].tail(SEQUENCE_LENGTH)
        
        if len(sample_window) == 0:
            st.info("No sensor data available for this engine/cycle combination.")
        else:
            fig_sensors, axes_sensors = plt.subplots(6, 4, figsize=(12, 8))
            axes_sensors = axes_sensors.flatten()
            
            for i, col in enumerate(sensor_cols):
                if i < len(axes_sensors):
                    axes_sensors[i].plot(sample_window[col].astype(float).values, linewidth=1.5, color='#667eea')
                    axes_sensors[i].set_title(f"S{i+1}", fontsize=8, pad=2)
                    axes_sensors[i].tick_params(labelsize=6)
                    axes_sensors[i].grid(True, alpha=0.2)
            
            # Hide extra subplots
            for i in range(len(sensor_cols), len(axes_sensors)):
                axes_sensors[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_sensors)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RIGHT COLUMN - EVALUATION & BATCH PROCESSING
# ============================================================================
with col_right:
    # ========================================================================
    # EVALUATION PLOT SECTION
    # ========================================================================
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Evaluation")
    
    # Display existing plot if available
    if os.path.exists(EVAL_PLOT):
        st.image(EVAL_PLOT, use_container_width=True, caption="RUL Prediction vs True RUL")
        
        # Try to show cached metrics if available
        if 'eval_metrics' in st.session_state:
            col_mae, col_rmse = st.columns(2)
            with col_mae:
                st.metric("MAE", f"{st.session_state.eval_metrics['mae']:.2f}", delta="cycles")
            with col_rmse:
                st.metric("RMSE", f"{st.session_state.eval_metrics['rmse']:.2f}", delta="cycles")
    else:
        st.info("📉 No evaluation plot found. Click 'Run Full Evaluation' to generate.")
    
    st.markdown("")  # Spacing
    
    if st.button("🔄 Run Full Evaluation", use_container_width=True):
        if df_test is None or scaler is None or model is None:
            st.error("⚠️ Missing required files: model, scaler, or test data.")
        else:
            with st.spinner("⏳ Running evaluate_model.py (this may take several minutes)..."):
                ok, out_text = run_evaluate_model(timeout_seconds=600)
            
            if not ok:
                st.error("❌ Evaluation failed or timed out.")
                with st.expander("🔍 Error Details"):
                    st.code(out_text[:4000], language="text")
            else:
                # Parse metrics
                mae, rmse = parse_metrics_from_output(out_text)
                
                if mae is not None and rmse is not None:
                    st.session_state.eval_metrics = {'mae': mae, 'rmse': rmse}
                
                with st.expander("📋 Evaluation Output"):
                    st.code(out_text[:8000], language="text")
                
                time.sleep(0.3)  # Allow file system to sync
                
                if os.path.exists(EVAL_PLOT):
                    st.success("✓ Evaluation complete!")
                    st.image(EVAL_PLOT, use_container_width=True)
                    
                    if mae is not None and rmse is not None:
                        col_mae, col_rmse = st.columns(2)
                        with col_mae:
                            st.metric("MAE", f"{mae:.2f}", delta="cycles")
                        with col_rmse:
                            st.metric("RMSE", f"{rmse:.2f}", delta="cycles")
                else:
                    st.error("❌ Plot file not found after evaluation. Check output above.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # BATCH EVALUATION SECTION
    # ========================================================================
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Batch Evaluation")
    st.caption("Run predictions on all test engines using final window of each engine")
    
    if st.button("▶️ Run Batch Evaluation", use_container_width=True):
        if df_test is None or scaler is None or model is None:
            st.error("⚠️ Missing required files for batch evaluation.")
        else:
            with st.spinner("⏳ Building final-window dataset and running predictions..."):
                X_seq_list, X_stats_list = [], []
                eids = sorted(df_test["engine_id"].unique())
                
                progress_bar = st.progress(0)
                for idx, eid in enumerate(eids):
                    edf = df_test[df_test["engine_id"] == eid].sort_values("cycle")
                    window = edf.tail(SEQUENCE_LENGTH)
                    
                    if len(window) < SEQUENCE_LENGTH:
                        pad_n = SEQUENCE_LENGTH - len(window)
                        pad_df = pd.DataFrame(np.repeat(window.iloc[[0]].values, pad_n, axis=0), columns=window.columns)
                        window = pd.concat([pad_df, window], ignore_index=True)
                    
                    seq = window[sensor_cols].astype(float).values
                    try:
                        seq_scaled = scaler.transform(seq)
                    except Exception:
                        seq_scaled = seq
                    X_seq_list.append(seq_scaled)
                    
                    s = window[sensor_cols]
                    stats = list(s.mean().fillna(0).values)
                    stats.extend(list(s.var(ddof=0).fillna(0).values))
                    stats.extend(list(s.skew().fillna(0).values))
                    stats.extend(list(s.kurt().fillna(0).values))
                    X_stats_list.append(stats)
                    
                    progress_bar.progress((idx + 1) / len(eids))
                
                progress_bar.empty()
                
                X_seq_test = np.array(X_seq_list).reshape(len(X_seq_list), SEQUENCE_LENGTH, len(sensor_cols))
                X_stats_test = np.array(X_stats_list)
                
                try:
                    y_pred = model.predict([X_seq_test, X_stats_test], verbose=0).flatten()
                except Exception:
                    y_pred = model.predict(X_seq_test, verbose=0).flatten()
                
                st.success(f"✓ Predicted RUL for {len(y_pred)} engines")
                
                # Calculate metrics if true RUL available
                if true_rul is not None and len(true_rul) == len(y_pred):
                    mae = np.mean(np.abs(true_rul - y_pred))
                    rmse = np.sqrt(np.mean((true_rul - y_pred) ** 2))
                    
                    col_batch_mae, col_batch_rmse = st.columns(2)
                    with col_batch_mae:
                        st.metric("Batch MAE", f"{mae:.2f}", delta="cycles")
                    with col_batch_rmse:
                        st.metric("Batch RMSE", f"{rmse:.2f}", delta="cycles")
                
                # Prediction distribution plot
                st.markdown("**Prediction Distribution**")
                fig_batch, ax_batch = plt.subplots(figsize=(10, 4))
                ax_batch.scatter(range(len(y_pred)), y_pred, alpha=0.6, s=30, color='#667eea')
                if true_rul is not None and len(true_rul) == len(y_pred):
                    ax_batch.scatter(range(len(true_rul)), true_rul, alpha=0.4, s=20, color='#ff6b6b', label='True RUL')
                    ax_batch.legend()
                ax_batch.set_xlabel("Test Engine Index")
                ax_batch.set_ylabel("RUL (cycles)")
                ax_batch.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_batch)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <small>
    💡 <strong>Note:</strong> Single-engine predictions use case_study.py to ensure consistent preprocessing.<br>
    Full evaluation runs evaluate_model.py which generates comprehensive metrics and plots.
    </small>
</div>
""", unsafe_allow_html=True)