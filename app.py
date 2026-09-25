import sys
import types
import os

# --- Windows Smart App Control compatibility shims for scikit-learn ---
if "sklearn.svm._liblinear" not in sys.modules:
    dummy_liblinear = types.ModuleType("sklearn.svm._liblinear")
    dummy_liblinear.set_verbosity_wrap = lambda x: None
    dummy_liblinear.train_wrap = lambda *args, **kwargs: (None, None)
    sys.modules["sklearn.svm._liblinear"] = dummy_liblinear

if "sklearn.ensemble._hist_gradient_boosting.gradient_boosting" not in sys.modules:
    dummy_hgb = types.ModuleType("sklearn.ensemble._hist_gradient_boosting.gradient_boosting")
    dummy_hgb.HistGradientBoostingClassifier = None
    dummy_hgb.HistGradientBoostingRegressor = None
    sys.modules["sklearn.ensemble._hist_gradient_boosting.gradient_boosting"] = dummy_hgb

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Sentinel: Hazardous Asteroid Prediction AI",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Futuristic Dark Styling ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
    }
    
    /* Header Card */
    .sentinel-header {
        background: linear-gradient(135deg, rgba(13, 22, 44, 0.9) 0%, rgba(20, 10, 35, 0.9) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .sentinel-title {
        color: #00e5ff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
    }
    
    .sentinel-sub {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Glowing Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 229, 255, 0.4);
    }
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00e5ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Result Banners */
    .alert-hazardous {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(153, 27, 27, 0.35) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        animation: pulseHazard 2s infinite ease-in-out;
    }
    
    @keyframes pulseHazard {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.25); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.5); }
    }
    
    .alert-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
    }
    
    .alert-title-haz {
        color: #ef4444;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    
    .alert-title-safe {
        color: #10b981;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: 2px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        color: #cbd5e1;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.95rem;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 229, 255, 0.15) !important;
        border-color: #00e5ff !important;
        color: #00e5ff !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# --- Artifact Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hazardous_asteroid_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "decision_threshold.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "neo__3_.csv")

# --- Resource Caching ---
@st.cache_resource
def load_ml_assets():
    """Load the trained machine learning pipeline artifacts."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(COLUMNS_PATH)
        default_threshold = float(joblib.load(THRESHOLD_PATH))
        return model, scaler, feature_cols, default_threshold, None
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data
def load_neo_data():
    """Load the NASA Near-Earth Object catalog for EDA and batch sample."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        # Drop unneeded columns for clean display
        drop_cols = [c for c in ['orbiting_body', 'sentry_object'] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df
    return None

model, scaler, feature_cols, default_threshold, load_err = load_ml_assets()

# --- Prediction Engine ---
def predict_asteroid_hazard(mag, d_min, d_max, vel, miss_dist, threshold):
    """Computes features, scales data, and performs model inference."""
    d_avg = (d_min + d_max) / 2.0
    ratio = vel / miss_dist if miss_dist > 0 else 0.0
    
    features_df = pd.DataFrame([[mag, d_avg, vel, miss_dist, ratio]], columns=feature_cols)
    scaled_features = scaler.transform(features_df)
    prob_hazardous = float(model.predict_proba(scaled_features)[0][1])
    is_hazardous = prob_hazardous >= threshold
    label = "Hazardous" if is_hazardous else "Non-Hazardous"
    
    # Kinetic energy proxy in Megatons of TNT (assumes spherical body, density 2.6 g/cm3)
    # E = 0.5 * m * v^2 ; 1 MT TNT = 4.184e15 Joules
    radius_m = (d_avg * 1000.0) / 2.0
    volume_m3 = (4.0 / 3.0) * np.pi * (radius_m ** 3)
    mass_kg = volume_m3 * 2600.0
    vel_ms = vel / 3.6
    energy_joules = 0.5 * mass_kg * (vel_ms ** 2)
    megatons_tnt = energy_joules / 4.184e15
    
    lunar_distances = miss_dist / 384400.0  # 1 LD = 384,400 km
    
    return {
        "label": label,
        "is_hazardous": is_hazardous,
        "probability": prob_hazardous,
        "d_avg": d_avg,
        "ratio": ratio,
        "megatons_tnt": megatons_tnt,
        "lunar_distances": lunar_distances,
        "velocity_kms": vel / 3600.0,
        "raw_features": features_df
    }

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### 🛰️ MISSION CONTROL")
    st.markdown("NASA Planetary Defense Coordination Office (PDCO) System")
    
    st.divider()
    
    st.markdown("#### ⚙️ Decision Threshold")
    st.markdown(
        "Threshold optimized via precision-recall tuning to maximize **Recall** for planetary defense."
    )
    
    custom_threshold = st.slider(
        "Hazard Probability Cutoff",
        min_value=0.05,
        max_value=0.95,
        value=float(default_threshold) if default_threshold else 0.40,
        step=0.05,
        help="Default 0.40 catches 74.8% of hazardous threats vs 57.1% at standard 0.50 threshold."
    )
    
    if custom_threshold != default_threshold:
        st.info(f"Custom threshold active: {custom_threshold:.2f} (Model Baseline: {default_threshold:.2f})")
    else:
        st.success(f"Running on Optimized Threshold: {default_threshold:.2f}")

    st.divider()
    
    st.markdown("#### 📐 Unit Reference")
    st.markdown("""
    - **1 Lunar Distance (LD)**: 384,400 km
    - **Tunguska Event**: ~10-15 MT TNT
    - **Chelyabinsk Meteor**: ~0.5 MT TNT
    - **Dinosaur Extinction (Chicxulub)**: ~100 Million MT
    """)
    
    st.divider()
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b;'>Project Sentinel v2.4 | Model: Tuned Random Forest (200 Trees, ROC-AUC: 0.9316)</div>",
        unsafe_allow_html=True
    )

# --- Top Banner Header ---
st.markdown("""
<div class="sentinel-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div class="sentinel-title">🛰️ PROJECT SENTINEL</div>
            <div class="sentinel-sub">Autonomous Near-Earth Object (NEO) Hazard Prediction & Planetary Defense AI</div>
        </div>
        <div style="display: flex; gap: 10px; margin-top: 10px;">
            <span style="background: rgba(0, 229, 255, 0.15); border: 1px solid #00e5ff; color: #00e5ff; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                DEFENSE STATUS: ACTIVE
            </span>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #10b981; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                ROC-AUC: 0.9316
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if load_err:
    st.error(f"⚠️ Error loading ML model assets: {load_err}")
    st.stop()

# --- Main App Navigation Tabs ---
tab_predict, tab_batch, tab_eda, tab_model = st.tabs([
    "🎯 Live Asteroid Predictor",
    "📂 Batch Analysis & Catalog",
    "📊 Exploratory Data Analysis (EDA)",
    "🧠 Model Performance & Explainability"
])

# ==============================================================================
# TAB 1: LIVE ASTEROID PREDICTOR
# ==============================================================================
with tab_predict:
    st.markdown("### 🎯 Real-Time Orbital Hazard Assessment")
    st.markdown("Input physical and trajectory characteristics of a Near Earth Object to evaluate impact risk.")
    
    # Quick preset buttons
    st.markdown("##### ⚡ Quick Scenarios")
    preset_cols = st.columns(4)
    
    # Store defaults in session state
    if "p_mag" not in st.session_state:
        st.session_state.p_mag = 18.0
        st.session_state.p_dmin = 0.12
        st.session_state.p_dmax = 0.28
        st.session_state.p_vel = 48000.0
        st.session_state.p_dist = 4500000.0

    if preset_cols[0].button("🛡️ Harmless Micro-Asteroid"):
        st.session_state.p_mag = 26.5
        st.session_state.p_dmin = 0.012
        st.session_state.p_dmax = 0.027
        st.session_state.p_vel = 14200.0
        st.session_state.p_dist = 52000000.0
        st.rerun()

    if preset_cols[1].button("⚠️ City-Threat Flyby (Apophis/Tunguska)"):
        st.session_state.p_mag = 17.5
        st.session_state.p_dmin = 0.35
        st.session_state.p_dmax = 0.78
        st.session_state.p_vel = 62000.0
        st.session_state.p_dist = 1800000.0
        st.rerun()

    if preset_cols[2].button("🚨 Chicxulub-Scale Hazard"):
        st.session_state.p_mag = 15.0
        st.session_state.p_dmin = 1.8
        st.session_state.p_dmax = 4.0
        st.session_state.p_vel = 88000.0
        st.session_state.p_dist = 850000.0
        st.rerun()

    if preset_cols[3].button("🌌 Typical NASA NEO Survey"):
        st.session_state.p_mag = 22.0
        st.session_state.p_dmin = 0.08
        st.session_state.p_dmax = 0.18
        st.session_state.p_vel = 34000.0
        st.session_state.p_dist = 18000000.0
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.1, 1.0], gap="large")
    
    with col_input:
        st.markdown("#### 🔭 Asteroid Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            in_mag = st.number_input(
                "Absolute Magnitude (H)",
                min_value=5.0,
                max_value=35.0,
                value=float(st.session_state.p_mag),
                step=0.25,
                help="Brightness proxy for size. Lower values mean significantly brighter/larger asteroids (e.g. H < 22 is critical for hazard classification)."
            )
        with c2:
            in_vel = st.number_input(
                "Relative Velocity (km/h)",
                min_value=500.0,
                max_value=250000.0,
                value=float(st.session_state.p_vel),
                step=1000.0,
                help="Speed of asteroid relative to Earth upon close approach."
            )
            
        c3, c4 = st.columns(2)
        with c3:
            in_dmin = st.number_input(
                "Estimated Diameter Min (km)",
                min_value=0.001,
                max_value=30.0,
                value=float(st.session_state.p_dmin),
                step=0.05,
                format="%.4f"
            )
        with c4:
            in_dmax = st.number_input(
                "Estimated Diameter Max (km)",
                min_value=0.002,
                max_value=60.0,
                value=float(st.session_state.p_dmax),
                step=0.05,
                format="%.4f"
            )
            
        in_dist = st.number_input(
            "Miss Distance from Earth (km)",
            min_value=5000.0,
            max_value=85000000.0,
            value=float(st.session_state.p_dist),
            step=100000.0,
            format="%.0f",
            help="Distance at closest approach. 1 Lunar Distance is ~384,400 km."
        )
        
        # Computed quick previews
        avg_diam_km = (in_dmin + in_dmax) / 2.0
        ratio_val = in_vel / in_dist if in_dist > 0 else 0
        lunar_eq = in_dist / 384400.0
        vel_kms = in_vel / 3600.0
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); padding: 12px; border-radius: 8px; margin-top: 10px; font-size: 0.9rem;">
            <b>Derived Physical Properties:</b><br>
            • Avg Diameter: <code>{:.3f} km</code> ({:.1f} m)<br>
            • Relative Velocity: <code>{:.2f} km/s</code> (Mach {:.0f})<br>
            • Miss Distance: <code>{:.2f} Lunar Distances (LD)</code><br>
            • Velocity-to-Distance Ratio: <code>{:.2e}</code>
        </div>
        """.format(avg_diam_km, avg_diam_km * 1000, vel_kms, vel_kms * 2.938, lunar_eq, ratio_val), unsafe_allow_html=True)

    # Compute prediction
    pred_res = predict_asteroid_hazard(in_mag, in_dmin, in_dmax, in_vel, in_dist, custom_threshold)

    with col_result:
        st.markdown("#### 🚨 Mission Assessment")
        
        # Verdict Card
        if pred_res["is_hazardous"]:
            st.markdown(f"""
            <div class="alert-hazardous">
                <div style="font-size: 2.5rem; margin-bottom: 5px;">⚠️ ☄️ 🚨</div>
                <div class="alert-title-haz">HAZARDOUS ASTEROID</div>
                <div style="color: #fca5a5; font-size: 1.1rem; margin-top: 6px;">
                    POTENTIAL IMPACT RISK DETECTED
                </div>
                <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                    Calculated Hazard Probability: <b>{pred_res['probability']:.1%}</b> (Threshold: {custom_threshold:.0%})
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-safe">
                <div style="font-size: 2.5rem; margin-bottom: 5px;">🛡️ 🌍 ✅</div>
                <div class="alert-title-safe">NON-HAZARDOUS</div>
                <div style="color: #6ee7b7; font-size: 1.1rem; margin-top: 6px;">
                    ORBITAL PATH SAFE FOR EARTH
                </div>
                <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                    Calculated Hazard Probability: <b>{pred_res['probability']:.1%}</b> (Threshold: {custom_threshold:.0%})
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge Chart for Probability
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_res["probability"] * 100,
            number={"suffix": "%", "font": {"family": "Orbitron", "size": 32, "color": "#00e5ff"}},
            title={"text": "Hazard Threat Probability", "font": {"size": 15, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#ef4444" if pred_res["is_hazardous"] else "#10b981", "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, custom_threshold * 100], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [custom_threshold * 100, 100], "color": "rgba(239, 68, 68, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "#facc15", "width": 3},
                    "thickness": 0.75,
                    "value": custom_threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(
            height=230,
            margin=dict(l=20, r=20, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Threat Details Metric Cards
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Impact Energy (Est.)</div>
                <div class="metric-value" style="font-size: 1.4rem;">{pred_res['megatons_tnt']:,.1f} MT</div>
                <div style="font-size: 0.75rem; color: #64748b;">TNT Equivalent Energy</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Close Approach</div>
                <div class="metric-value" style="font-size: 1.4rem;">{pred_res['lunar_distances']:.2f} LD</div>
                <div style="font-size: 0.75rem; color: #64748b;">({pred_res['lunar_distances']*384400:,.0f} km)</div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: BATCH ANALYSIS & CATALOG INSPECTOR
# ==============================================================================
with tab_batch:
    st.markdown("### 📂 Batch Inference & NASA Catalog Exploration")
    st.markdown("Analyze entire CSV batches of Near-Earth Objects or sample directly from NASA's 90,836 NEO dataset.")
    
    col_upload, col_sample = st.columns([1.2, 1.0])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Custom Asteroid CSV (Must include columns: absolute_magnitude, est_diameter_min, est_diameter_max, relative_velocity, miss_distance)",
            type=["csv"]
        )
    with col_sample:
        st.markdown("##### 🚀 Quick Demo Samples")
        sample_size = st.slider("Sample Size from NASA NEO Dataset", 10, 200, 50, step=10)
        use_nasa_sample = st.button("📥 Load Random Samples from Catalog", type="primary")

    df_to_analyze = None
    df_raw = load_neo_data()
    
    if uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(df_to_analyze)} objects.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif use_nasa_sample or "current_sample_df" in st.session_state:
        if use_nasa_sample and df_raw is not None:
            st.session_state.current_sample_df = df_raw.sample(n=min(sample_size, len(df_raw)), random_state=np.random.randint(1, 9999))
        if "current_sample_df" in st.session_state:
            df_to_analyze = st.session_state.current_sample_df.copy()

    if df_to_analyze is not None:
        # Check required columns
        req_cols = ['absolute_magnitude', 'est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance']
        missing = [c for c in req_cols if c not in df_to_analyze.columns]
        
        if missing:
            st.error(f"Uploaded dataset is missing required features: {missing}")
        else:
            with st.spinner("Processing batch feature engineering and model inference..."):
                df_calc = df_to_analyze.copy()
                df_calc['est_diameter_avg'] = (df_calc['est_diameter_min'] + df_calc['est_diameter_max']) / 2.0
                df_calc['velocity_distance_ratio'] = df_calc['relative_velocity'] / df_calc['miss_distance']
                
                # Transform
                X_batch = df_calc[feature_cols]
                X_batch_scaled = scaler.transform(X_batch)
                
                # Predict
                probs = model.predict_proba(X_batch_scaled)[:, 1]
                preds = (probs >= custom_threshold).astype(int)
                
                df_calc['Hazard_Probability'] = np.round(probs, 4)
                df_calc['Hazard_Status'] = np.where(preds == 1, '🚨 Hazardous', '✅ Safe')
                df_calc['Miss_Distance_LD'] = np.round(df_calc['miss_distance'] / 384400.0, 2)
            
            # KPI Metrics
            total_objs = len(df_calc)
            haz_count = int(np.sum(preds))
            safe_count = total_objs - haz_count
            haz_rate = (haz_count / total_objs) * 100.0 if total_objs > 0 else 0
            
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            b_c1.metric("Total Analyzed", f"{total_objs:,}")
            b_c2.metric("Hazardous Detected", f"{haz_count:,}", delta=f"{haz_rate:.1f}% Threat Rate", delta_color="inverse")
            b_c3.metric("Safe NEOs", f"{safe_count:,}")
            b_c4.metric("Avg Velocity", f"{df_calc['relative_velocity'].mean():,.0f} km/h")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive 2D/3D Scatter Plot
            st.markdown("#### 🌌 Orbital Threat Distribution")
            scatter_fig = px.scatter(
                df_calc,
                x="miss_distance",
                y="relative_velocity",
                size="est_diameter_avg",
                color="Hazard_Status",
                color_discrete_map={'🚨 Hazardous': '#ef4444', '✅ Safe': '#10b981'},
                hover_data=['absolute_magnitude', 'Hazard_Probability', 'Miss_Distance_LD'],
                labels={
                    "miss_distance": "Miss Distance (km)",
                    "relative_velocity": "Relative Velocity (km/h)",
                    "est_diameter_avg": "Avg Diameter (km)"
                },
                title="Miss Distance vs. Relative Velocity (Size = Diameter)",
                template="plotly_dark"
            )
            scatter_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                legend_title_text="Predicted Status"
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Data Table Preview & Download
            st.markdown("#### 📋 Prediction Registry")
            display_cols = [c for c in ['id', 'name', 'absolute_magnitude', 'est_diameter_avg', 'relative_velocity', 'Miss_Distance_LD', 'Hazard_Probability', 'Hazard_Status'] if c in df_calc.columns]
            st.dataframe(df_calc[display_cols].sort_values("Hazard_Probability", ascending=False), use_container_width=True)
            
            # Download Button
            csv_export = df_calc.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results as CSV",
                data=csv_export,
                file_name="sentinel_asteroid_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Click **'Load Random Samples from Catalog'** or upload your own CSV file above to test batch evaluation.")

# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("### 📊 NASA Near-Earth Object Dataset Architecture")
    st.markdown("Detailed exploration of the 90,836 NEO astronomical catalog used to train and validate Sentinel AI.")
    
    df_eda = load_neo_data()
    if df_eda is not None:
        e_c1, e_c2, e_c3 = st.columns(3)
        e_c1.metric("Total Catalog Entries", f"{len(df_eda):,}")
        e_c2.metric("True Hazardous Count", f"{df_eda['hazardous'].sum():,} ({df_eda['hazardous'].mean():.1%})")
        e_c3.metric("Non-Hazardous Count", f"{(~df_eda['hazardous']).sum():,} ({(~df_eda['hazardous']).mean():.1%})")
        
        st.divider()
        
        # Donut Chart and Distribution
        col_donut, col_dist = st.columns([1.0, 1.3])
        
        with col_donut:
            st.markdown("#### ⚖️ Severe Class Imbalance")
            class_counts = df_eda['hazardous'].value_counts()
            donut_fig = go.Figure(data=[go.Pie(
                labels=['Non-Hazardous (Safe)', 'Hazardous (Threat)'],
                values=class_counts.values,
                hole=.55,
                marker_colors=['#0ea5e9', '#ef4444'],
                textinfo='label+percent'
            )])
            donut_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300
            )
            st.plotly_chart(donut_fig, use_container_width=True)
            st.caption("Only ~9.7% of observed NEOs are classified as hazardous. This motivated the use of `class_weight='balanced_subsample'` and threshold tuning.")

        with col_dist:
            st.markdown("#### 🔭 Absolute Magnitude (H) vs Hazard")
            hist_fig = px.histogram(
                df_eda.sample(min(10000, len(df_eda)), random_state=42),
                x="absolute_magnitude",
                color="hazardous",
                barmode="overlay",
                color_discrete_map={False: '#0ea5e9', True: '#ef4444'},
                labels={"absolute_magnitude": "Absolute Magnitude (H)", "hazardous": "Hazardous"},
                template="plotly_dark",
                nbins=40
            )
            hist_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(hist_fig, use_container_width=True)
            st.caption("Notice: Hazardous asteroids are concentrated at lower H magnitudes (< 22), indicating larger optical brightness and physical volume.")

        st.divider()

        # Correlation and Physical Separation
        col_box, col_scatter = st.columns(2)
        
        with col_box:
            st.markdown("#### ⚡ Relative Velocity Comparison")
            box_fig = px.box(
                df_eda.sample(min(10000, len(df_eda)), random_state=42),
                x="hazardous",
                y="relative_velocity",
                color="hazardous",
                color_discrete_map={False: '#0ea5e9', True: '#ef4444'},
                labels={"hazardous": "Is Hazardous", "relative_velocity": "Relative Velocity (km/h)"},
                template="plotly_dark"
            )
            box_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                showlegend=False,
                height=320
            )
            st.plotly_chart(box_fig, use_container_width=True)

        with col_scatter:
            st.markdown("#### 📏 Miss Distance vs Diameter")
            df_eda_sample = df_eda.sample(min(8000, len(df_eda)), random_state=42).copy()
            df_eda_sample['avg_diam'] = (df_eda_sample['est_diameter_min'] + df_eda_sample['est_diameter_max']) / 2.0
            
            sc_fig = px.scatter(
                df_eda_sample,
                x="miss_distance",
                y="avg_diam",
                color="hazardous",
                color_discrete_map={False: '#0ea5e9', True: '#ef4444'},
                labels={"miss_distance": "Miss Distance (km)", "avg_diam": "Diameter (km)", "hazardous": "Hazardous"},
                template="plotly_dark",
                opacity=0.6
            )
            sc_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(sc_fig, use_container_width=True)

# ==============================================================================
# TAB 4: MODEL PERFORMANCE & EXPLAINABILITY
# ==============================================================================
with tab_model:
    st.markdown("### 🧠 Machine Learning Architecture & Benchmark")
    st.markdown("Evaluation metrics, threshold optimization curves, and feature importance rankings for planetary defense.")
    
    # Model Comparison Table
    st.markdown("#### 🏆 Model Comparison on 18,168 Test NEOs")
    comparison_data = {
        "Model": [
            "Random Forest (Tuned + 0.40 Thresh)",
            "Random Forest (Default)",
            "Decision Tree (Balanced)",
            "Logistic Regression (Balanced)",
            "Gradient Boosting"
        ],
        "Accuracy": [0.8802, 0.9150, 0.7962, 0.7861, 0.9150],
        "Precision": [0.4302, 0.6176, 0.3195, 0.3043, 0.8159],
        "Recall": [0.7483, 0.3326, 0.9689, 0.9321, 0.1629],
        "F1-Score": [0.5464, 0.4324, 0.4806, 0.4589, 0.2716],
        "ROC-AUC": [0.9316, 0.9313, 0.9046, 0.8795, 0.9212]
    }
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(
        df_comp.style.highlight_max(subset=["Recall", "F1-Score", "ROC-AUC"], color="#065f46"),
        use_container_width=True
    )
    
    st.divider()

    # Threshold Optimization Curve & Confusion Matrix
    col_thresh_curve, col_cm = st.columns(2)
    
    with col_thresh_curve:
        st.markdown("#### 📈 Threshold Tuning Dynamics")
        threshold_curve_data = pd.DataFrame({
            "Threshold": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
            "Precision": [0.317, 0.327, 0.341, 0.356, 0.377, 0.401, 0.430, 0.456, 0.490, 0.524],
            "Recall":    [0.987, 0.972, 0.950, 0.910, 0.869, 0.816, 0.748, 0.663, 0.571, 0.484],
            "F1-Score":  [0.480, 0.489, 0.501, 0.512, 0.526, 0.537, 0.546, 0.540, 0.528, 0.503]
        })
        
        tc_fig = go.Figure()
        tc_fig.add_trace(go.Scatter(x=threshold_curve_data["Threshold"], y=threshold_curve_data["Recall"], mode='lines+markers', name='Recall', line=dict(color='#ef4444', width=3)))
        tc_fig.add_trace(go.Scatter(x=threshold_curve_data["Threshold"], y=threshold_curve_data["Precision"], mode='lines+markers', name='Precision', line=dict(color='#0ea5e9', width=2)))
        tc_fig.add_trace(go.Scatter(x=threshold_curve_data["Threshold"], y=threshold_curve_data["F1-Score"], mode='lines+markers', name='F1-Score', line=dict(color='#10b981', width=3)))
        
        tc_fig.add_vline(x=0.40, line_dash="dash", line_color="#facc15", annotation_text="Tuned Opt: 0.40", annotation_position="top left")
        tc_fig.update_layout(
            title="Recall vs. Precision vs. F1 Across Decision Thresholds",
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=340
        )
        st.plotly_chart(tc_fig, use_container_width=True)

    with col_cm:
        st.markdown("#### 🎯 Test Set Confusion Matrix (@ Thresh 0.40)")
        cm_matrix = np.array([[14646, 1754], [445, 1323]])
        cm_fig = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Class", y="True Ground Truth", color="NEO Count"),
            x=['Non-Hazardous', 'Hazardous'],
            y=['Non-Hazardous', 'Hazardous'],
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    st.divider()

    # Feature Importance
    st.markdown("#### 🔍 Random Forest Feature Importance (MDI / Gini)")
    feat_imp = pd.DataFrame({
        "Feature": [
            "absolute_magnitude (H)",
            "est_diameter_avg (km)",
            "relative_velocity (km/h)",
            "miss_distance (km)",
            "velocity_distance_ratio"
        ],
        "Importance": [0.3810, 0.3473, 0.1046, 0.0839, 0.0831]
    }).sort_values("Importance", ascending=True)

    fi_fig = px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        text_auto=".2%"
    )
    fi_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        height=280
    )
    st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown("""
    > [!IMPORTANT]
    > **Planetary Defense Rationale**: In astronomical threat interception, a **False Negative** (failing to detect an asteroid that hits Earth) is catastrophic, whereas a **False Positive** (monitoring a benign asteroid) costs only telescope observation time. Tuning the decision threshold to `0.40` boosted hazard recall to **74.8%** while maintaining a solid **0.9316 ROC-AUC**.
    """)

# --- Footer ---
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.85rem; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.06);">
    🛰️ NASA Planetary Defense Coordination Office Capstone • Built with Streamlit, Scikit-Learn & Plotly
</div>
""", unsafe_allow_html=True)
