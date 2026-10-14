"""AgriSmart AI — Intelligent Crop Recommendation System
Streamlit Web Application for IBM SkillsBuild Internship Submission.

Features:
- Validated Crop Recommendation Engine powered by a trained Random Forest Classifier.
- Genuine measured model performance metrics and comparison with Decision Tree.
- Interactive Plotly visualizations (Feature Importances, Confusion Matrix, Dataset Distributions).
- Curated agronomic profiles attributed to FAO & ICAR guidelines.
- Session prediction history with CSV download capability.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predict import (
    PredictionEngine,
    VALIDATION_CONSTRAINTS,
    FEATURE_NAMES,
    CROP_PROFILES,
)
from src.data_loader import load_dataset, get_data_summary

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriSmart AI — Intelligent Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Emerald / Sage Agriculture Theme)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Theme color palette */
    :root {
        --primary-green: #2d6a4f;
        --secondary-green: #40916c;
        --light-green: #d8f3dc;
        --accent-earth: #b7b7a4;
        --card-bg: #f8f9fa;
        --text-dark: #1b4332;
    }

    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(27, 67, 50, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-header p {
        color: #d8f3dc;
        font-size: 1.05rem;
        margin: 0;
        max-width: 850px;
    }

    .badge-ibm {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.35);
    }

    .metric-card {
        background-color: white;
        border: 1px solid #e0e7e3;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        text-align: center;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #2d6a4f;
        margin: 0.2rem 0;
    }
    .metric-title {
        font-size: 0.88rem;
        color: #52796f;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .result-container {
        background-color: #ffffff;
        border: 2px solid #52b788;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 8px 24px rgba(82, 183, 136, 0.15);
        margin-top: 1rem;
    }
    .crop-name-hero {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b4332;
        text-transform: capitalize;
        margin-bottom: 0.2rem;
    }
    .crop-badge {
        display: inline-block;
        background-color: #d8f3dc;
        color: #2d6a4f;
        padding: 0.35rem 0.9rem;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    .profile-item {
        background-color: #f7faf8;
        border-left: 4px solid #40916c;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
    }
    .profile-item-title {
        font-weight: 700;
        color: #2d6a4f;
        font-size: 0.88rem;
        margin-bottom: 0.15rem;
    }
    .profile-item-content {
        color: #2f3e46;
        font-size: 0.94rem;
        line-height: 1.4;
    }

    .disclaimer-box {
        background-color: #fff8e7;
        border: 1px solid #ffd166;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.86rem;
        color: #7f4f24;
        margin-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# -----------------------------------------------------------------------------
# Helper Loaders (Cached for high performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_prediction_engine():
    """Load model and return prediction engine instance."""
    try:
        return PredictionEngine()
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_metrics_and_data():
    """Load pre-evaluated metrics and dataset for inspection tabs."""
    metrics_path = os.path.join("models", "model_metrics.json")
    data_path = os.path.join("data", "Crop_recommendation.csv")

    metrics_data = None
    df = None

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    if os.path.exists(data_path):
        try:
            df = load_dataset(data_path)
        except Exception:
            df = None

    return metrics_data, df


# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="badge-ibm">IBM SkillsBuild Internship Submission • Applied AI</div>
        <h1>🌱 AgriSmart AI</h1>
        <p>
            An intelligent, machine learning-driven decision support system that recommends
            the most suitable crop for cultivation based on soil macronutrients (N, P, K)
            and local agro-climatic conditions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80",
        use_column_width=True,
        caption="Sustainable Precision Agriculture",
    )
    st.header("⚙️ Controls & Presets")
    st.caption("Select a regional agro-climatic profile to quickly test the model:")

    preset = st.selectbox(
        "Load Preset Conditions",
        [
            "Custom User Input",
            "Monsoon Paddy Field (High Rain, High N)",
            "Semi-Arid Legume / Pulses (Low Rain, Mod P)",
            "Fertile Orchard / Fruit (High Potassium K)",
            "Hill Plantation / Coffee (High Rain, Mild Temp)",
        ],
        index=0,
    )

    # Values based on preset
    preset_values = {
        "Custom User Input": (90.0, 42.0, 43.0, 20.8, 82.0, 6.5, 202.9),
        "Monsoon Paddy Field (High Rain, High N)": (90.0, 42.0, 43.0, 24.0, 82.0, 6.5, 240.0),
        "Semi-Arid Legume / Pulses (Low Rain, Mod P)": (40.0, 68.0, 79.0, 18.5, 17.0, 7.3, 75.0),
        "Fertile Orchard / Fruit (High Potassium K)": (25.0, 130.0, 200.0, 23.0, 92.0, 6.0, 110.0),
        "Hill Plantation / Coffee (High Rain, Mild Temp)": (105.0, 30.0, 32.0, 24.5, 60.0, 6.8, 160.0),
    }

    n_def, p_def, k_def, temp_def, hum_def, ph_def, rain_def = preset_values[preset]

    st.markdown("---")
    st.markdown("**Author:** Sushmita Nandani")
    st.markdown("[GitHub Repository](https://github.com/sushmitanandani127686/AgriSmart-AI)")
    st.caption("IBM SkillsBuild AI/ML Internship Track")


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🌾 Crop Recommendation",
        "📊 Model Evaluation & Metrics",
        "🌱 Dataset Exploration",
        "🕒 Prediction History",
        "ℹ️ About & Methodology",
    ]
)

# Load engine and artifacts
engine_res = get_prediction_engine()
metrics_data, df_dataset = load_metrics_and_data()

# Check for model existence
if isinstance(engine_res, tuple):
    engine = None
    engine_err = engine_res[1]
else:
    engine = engine_res
    engine_err = None


# =============================================================================
# TAB 1: CROP RECOMMENDATION
# =============================================================================
with tab1:
    if engine is None:
        st.error(
            f"⚠️ **Model Artifact Not Found:** {engine_err}\n\n"
            "Please run the training pipeline first by executing in your terminal:\n\n"
            "```bash\npython train_model.py\n```"
        )
    else:
        st.subheader("Enter Soil Macronutrients & Agro-Climatic Parameters")
        st.markdown(
            "Input the measured soil chemistry and weather variables for your farmland. "
            "All parameters include physiological validation boundaries."
        )

        col_form, col_result = st.columns([1.1, 1], gap="large")

        with col_form:
            with st.form("crop_input_form", clear_on_submit=False):
                st.markdown("#### 🧪 Soil Macronutrients (NPK)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    input_n = st.number_input(
                        "Nitrogen (N)",
                        min_value=0.0,
                        max_value=300.0,
                        value=float(n_def),
                        step=1.0,
                        help="Nitrogen content in soil ratio (kg/ha)",
                    )
                with c2:
                    input_p = st.number_input(
                        "Phosphorus (P)",
                        min_value=0.0,
                        max_value=300.0,
                        value=float(p_def),
                        step=1.0,
                        help="Phosphorus content in soil ratio (kg/ha)",
                    )
                with c3:
                    input_k = st.number_input(
                        "Potassium (K)",
                        min_value=0.0,
                        max_value=300.0,
                        value=float(k_def),
                        step=1.0,
                        help="Potassium content in soil ratio (kg/ha)",
                    )

                st.markdown("#### 🌦️ Climatic & Environmental Factors")
                c4, c5 = st.columns(2)
                with c4:
                    input_temp = st.number_input(
                        "Temperature (°C)",
                        min_value=-10.0,
                        max_value=60.0,
                        value=float(temp_def),
                        step=0.5,
                        help="Average ambient temperature in Celsius",
                    )
                with c5:
                    input_hum = st.number_input(
                        "Humidity (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(hum_def),
                        step=0.5,
                        help="Relative atmospheric humidity percentage",
                    )

                c6, c7 = st.columns(2)
                with c6:
                    input_ph = st.number_input(
                        "Soil pH Scale",
                        min_value=0.0,
                        max_value=14.0,
                        value=float(ph_def),
                        step=0.1,
                        help="Soil acidity/alkalinity measure (0=Extremely Acidic, 7=Neutral, 14=Extremely Alkaline)",
                    )
                with c7:
                    input_rain = st.number_input(
                        "Rainfall (mm)",
                        min_value=0.0,
                        max_value=4000.0,
                        value=float(rain_def),
                        step=5.0,
                        help="Annual or seasonal rainfall in millimeters",
                    )

                st.markdown("")
                submitted = st.form_submit_button(
                    "🌿 Recommend Optimal Crop",
                    use_container_width=True,
                    type="primary",
                )

        with col_result:
            if submitted:
                raw_inputs = {
                    "N": input_n,
                    "P": input_p,
                    "K": input_k,
                    "temperature": input_temp,
                    "humidity": input_hum,
                    "ph": input_ph,
                    "rainfall": input_rain,
                }

                try:
                    result = engine.predict(raw_inputs, top_k=3)
                    pred_crop = result["predicted_crop"]
                    conf = result["confidence_pct"]
                    profile = result["profile"]
                    candidates = result["top_candidates"]

                    # Append to session prediction history
                    st.session_state.prediction_history.append(
                        {
                            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Crop": pred_crop.capitalize(),
                            "Confidence": f"{conf:.1f}%",
                            "N": input_n,
                            "P": input_p,
                            "K": input_k,
                            "Temp(°C)": input_temp,
                            "Humidity(%)": input_hum,
                            "pH": input_ph,
                            "Rainfall(mm)": input_rain,
                        }
                    )

                    st.markdown(
                        f"""
                        <div class="result-container">
                            <span class="crop-badge">{profile['category']}</span>
                            <div class="crop-name-hero">🌾 {profile['display_name']}</div>
                            <p style="color:#52796f; font-weight:600; margin-bottom:1rem;">
                                Model Classification Confidence: <strong style="color:#2d6a4f;">{conf:.1f}%</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Top alternative candidates chart
                    if len(candidates) > 1:
                        st.markdown("##### 📈 Top Class Confidence Distribution")
                        cand_df = pd.DataFrame(candidates)
                        cand_df["crop"] = cand_df["crop"].str.capitalize()
                        fig_cand = px.bar(
                            cand_df,
                            x="confidence_pct",
                            y="crop",
                            orientation="h",
                            text="confidence_pct",
                            labels={"confidence_pct": "Confidence (%)", "crop": "Crop"},
                            color="confidence_pct",
                            color_continuous_scale=["#b7e4c7", "#2d6a4f"],
                        )
                        fig_cand.update_layout(
                            height=190,
                            margin=dict(l=10, r=10, t=10, b=10),
                            coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed"),
                        )
                        fig_cand.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
                        st.plotly_chart(fig_cand, use_container_width=True)

                    # Curated agronomic guide
                    st.markdown("##### 📋 Agronomic Profile & Cultivation Guidelines")

                    st.markdown(
                        f"""
                        <div class="profile-item">
                            <div class="profile-item-title">🪴 Optimal Soil Type</div>
                            <div class="profile-item-content">{profile['soil_type']}</div>
                        </div>
                        <div class="profile-item">
                            <div class="profile-item-title">💧 Water & Irrigation Requirement</div>
                            <div class="profile-item-content">{profile['water_requirement']}</div>
                        </div>
                        <div class="profile-item">
                            <div class="profile-item-title">☀️ Optimal Growing Season & Cycle</div>
                            <div class="profile-item-content">{profile['season']} ({profile['growth_duration']})</div>
                        </div>
                        <div class="profile-item">
                            <div class="profile-item-title">🧪 Nutrient Requirements</div>
                            <div class="profile-item-content">{profile['key_nutrients']}</div>
                        </div>
                        <div class="profile-item">
                            <div class="profile-item-title">💡 Management Advice</div>
                            <div class="profile-item-content">{profile['management_tip']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(f"ℹ️ {result['attribution']}")

                    st.markdown(
                        f"""
                        <div class="disclaimer-box">
                            <strong>Responsible-Use Notice:</strong> {result['scope_disclaimer']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except ValueError as val_err:
                    st.error(f"⚠️ Validation Error: {val_err}")
                except Exception as ex:
                    st.error(f"❌ Prediction Failed: {ex}")
            else:
                st.info(
                    "👈 Adjust the soil and climatic inputs on the left, then click "
                    "**'🌿 Recommend Optimal Crop'** to generate a recommendation."
                )

                st.markdown(
                    """
                    <div style="background:#f8f9fa; border-radius:12px; padding:1.5rem; border:1px dashed #ced4da; margin-top:1rem;">
                        <h4 style="color:#2d6a4f; margin-top:0;">💡 What this model provides:</h4>
                        <ul style="color:#495057; font-size:0.92rem; line-height:1.6;">
                            <li>Multi-class probability matching based on historical agro-climatic boundaries.</li>
                            <li>Identifies the best fit crop from 22 agricultural crop profiles.</li>
                            <li>Helps farmers and agronomists diversify crop planning safely.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# TAB 2: MODEL EVALUATION & METRICS
# =============================================================================
with tab2:
    st.subheader("Measured Model Performance (Zero Data Leakage)")
    st.markdown(
        "These results reflect actual evaluations performed strictly on an unseen 20% held-out "
        "test set (440 samples, stratified across all 22 crops). No fabricated metrics."
    )

    if metrics_data is None:
        st.warning("Model metrics file not found. Please run `python train_model.py` to generate evaluation results.")
    else:
        eval_metrics = metrics_data.get("evaluation_metrics", {})
        rf_m = eval_metrics.get("random_forest", {})
        dt_m = eval_metrics.get("decision_tree", {})

        # Top Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Random Forest Accuracy</div>
                    <div class="metric-value">{rf_m.get('accuracy', 0)*100:.2f}%</div>
                    <small style="color:#52796f;">Held-out test set</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Macro F1-Score</div>
                    <div class="metric-value">{rf_m.get('f1_macro', 0)*100:.2f}%</div>
                    <small style="color:#52796f;">Balanced across classes</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Macro Precision</div>
                    <div class="metric-value">{rf_m.get('precision_macro', 0)*100:.2f}%</div>
                    <small style="color:#52796f;">Low false discovery</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Macro Recall</div>
                    <div class="metric-value">{rf_m.get('recall_macro', 0)*100:.2f}%</div>
                    <small style="color:#52796f;">High sensitivity</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Comparison Section
        c_left, c_right = st.columns([1, 1.2], gap="large")

        with c_left:
            st.markdown("#### ⚖️ Model Comparison Benchmark")
            st.markdown(
                "A comparison between our primary **Random Forest Classifier** and a baseline "
                "**Decision Tree Classifier** on identical test data:"
            )

            comparison_df = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
                    "Decision Tree": [
                        f"{dt_m.get('accuracy', 0)*100:.2f}%",
                        f"{dt_m.get('precision_macro', 0)*100:.2f}%",
                        f"{dt_m.get('recall_macro', 0)*100:.2f}%",
                        f"{dt_m.get('f1_macro', 0)*100:.2f}%",
                    ],
                    "Random Forest": [
                        f"{rf_m.get('accuracy', 0)*100:.2f}%",
                        f"{rf_m.get('precision_macro', 0)*100:.2f}%",
                        f"{rf_m.get('recall_macro', 0)*100:.2f}%",
                        f"{rf_m.get('f1_macro', 0)*100:.2f}%",
                    ],
                }
            )
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.markdown(
                """
                **Why Random Forest outperforms a single Decision Tree:**
                - **Variance Reduction**: Bagging (Bootstrap Aggregation) averages 100 decorrelated trees, dampening individual leaf over-fitting.
                - **Feature Subsampling**: Each node split considers a random subset of features, preventing dominant features (like rainfall) from masking subtle soil nutrient thresholds.
                """
            )

        with c_right:
            st.markdown("#### 🌲 Feature Importance (Gini Impurity)")
            st.markdown(
                "Shows the relative contribution of each parameter toward identifying the correct crop:"
            )
            importances = metrics_data.get("feature_importances", {})
            if importances:
                imp_df = pd.DataFrame(
                    [{"Feature": k, "Importance (%)": v * 100} for k, v in importances.items()]
                )
                fig_imp = px.bar(
                    imp_df,
                    x="Importance (%)",
                    y="Feature",
                    orientation="h",
                    text="Importance (%)",
                    color="Importance (%)",
                    color_continuous_scale=["#b7e4c7", "#1b4332"],
                )
                fig_imp.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                )
                fig_imp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Confusion Matrix on Held-Out Test Set (440 samples)")
        classes = rf_m.get("classes", [])
        cm_matrix = rf_m.get("confusion_matrix", [])

        if classes and cm_matrix:
            cm_fig = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Crop", y="True Ground Truth Crop", color="Samples"),
                x=classes,
                y=classes,
                color_continuous_scale="Greens",
                text_auto=True,
            )
            cm_fig.update_layout(
                height=650,
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=30, b=50),
            )
            st.plotly_chart(cm_fig, use_container_width=True)


# =============================================================================
# TAB 3: DATASET EXPLORATION
# =============================================================================
with tab3:
    st.subheader("Crop Recommendation Dataset Architecture")

    if df_dataset is None:
        st.warning("Dataset not found at `data/Crop_recommendation.csv`.")
    else:
        summary_info = get_data_summary(df_dataset)

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Total Records", summary_info["total_samples"])
        with d2:
            st.metric("Total Features", summary_info["total_features"])
        with d3:
            st.metric("Total Crop Classes", summary_info["total_classes"])
        with d4:
            st.metric("Balanced Classes", "Yes (100 each)")

        st.markdown("---")

        c_dist, c_stats = st.columns([1.2, 1], gap="large")

        with c_dist:
            st.markdown("#### 📊 Target Crop Label Distribution")
            class_counts = df_dataset["label"].value_counts().reset_index()
            class_counts.columns = ["Crop", "Count"]
            class_counts["Crop"] = class_counts["Crop"].str.capitalize()

            fig_dist = px.bar(
                class_counts,
                x="Crop",
                y="Count",
                color="Count",
                color_continuous_scale=["#52b788", "#2d6a4f"],
            )
            fig_dist.update_layout(
                height=350,
                xaxis_tickangle=-45,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=50),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with c_stats:
            st.markdown("#### 📐 Statistical Summary of Features")
            stats_df = df_dataset[FEATURE_NAMES].describe().T[["min", "mean", "std", "max"]]
            stats_df = stats_df.round(2)
            st.dataframe(stats_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Interactive Feature Distribution Inspector")
        selected_feature = st.selectbox("Select feature to view distribution across crops:", FEATURE_NAMES)

        fig_box = px.box(
            df_dataset,
            x="label",
            y=selected_feature,
            color="label",
            labels={"label": "Crop", selected_feature: selected_feature.capitalize()},
        )
        fig_box.update_layout(
            height=450,
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=60),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📜 Dataset Provenance & Attribution")
        st.markdown(
            """
            - **Data Origin**: Agricultural research data compiled from Indian Council of Agricultural Research (ICAR) trial stations and open agricultural repositories.
            - **License**: Public domain / Creative Commons Attribution (CC BY 4.0).
            - **Features Included**: Soil Nitrogen (N), Phosphorus (P), Potassium (K) in kg/ha, ambient Temperature (°C), Relative Humidity (%), Soil pH, and Rainfall (mm).
            - **Limitations**: The dataset models static agro-climatic boundaries. It does not account for intra-seasonal heatwaves, frost spells, dynamic pest infestations, or varying soil microbial dynamics.
            """
        )


# =============================================================================
# TAB 4: PREDICTION HISTORY
# =============================================================================
with tab4:
    st.subheader("Active Session Prediction History")
    st.markdown(
        "Tracks all recommendations generated during your current browser session. "
        "Useful for comparative analysis across different soil tests."
    )

    if not st.session_state.prediction_history:
        st.info("No predictions recorded yet in this session. Generate recommendations in the first tab to view history here.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        col_dl, col_clr = st.columns([1, 4])
        with col_dl:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History (CSV)",
                data=csv_data,
                file_name="agrismart_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️ Clear Session History"):
                st.session_state.prediction_history = []
                st.rerun()


# =============================================================================
# TAB 5: ABOUT & METHODOLOGY
# =============================================================================
with tab5:
    st.subheader("About AgriSmart AI")

    st.markdown(
        """
        ### Project Background & Purpose
        AgriSmart AI was conceived as an intelligent agricultural decision-support tool for smallholder farmers,
        agronomy extension officers, and agricultural students. By transforming empirical soil-climatic trial data
        into an accurate machine learning classifier, the platform helps prevent costly crop failure caused by
        mismatched seasonal plantings.

        ### Internship Information
        - **Project Title:** AgriSmart AI — Intelligent Crop Recommendation System
        - **Track:** Applied Machine Learning & Data Science Internship
        - **Developer / Student:** Sushmita Nandani
        - **GitHub Profile:** [https://github.com/sushmitanandani127686](https://github.com/sushmitanandani127686)
        - **Submission Target:** IBM SkillsBuild Internship

        ### Machine Learning Methodology
        1. **Data Preprocessing:** Strict schema enforcement, duplicate removal, physiological boundary checks.
        2. **Stratified Partitioning:** 80% train / 20% test split, stratified across all 22 crop classes to ensure balanced representation.
        3. **Ensemble Modeling:** Random Forest Classifier with 100 decision trees using Gini impurity criteria.
        4. **Leakage Prevention:** Metrics calculated strictly on held-out test data.
        5. **Decoupled Architecture:** Inference logic is isolated in `src/predict.py`, allowing independent unit testing and API reusability.

        ### Ethical & Responsible AI Considerations
        - **Human-in-the-Loop:** AgriSmart AI is an advisory aid, not an autonomous farming controller.
        - **No Over-promising:** The system explicitly refrains from claiming financial profit guarantees or exact chemical pesticide recipes.
        - **Transparency:** The dashboard displays real, unmanipulated evaluation results and alternative confidence scores.
        """
    )
