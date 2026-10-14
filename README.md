# AgriSmart AI — Intelligent Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-v1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.39.1-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-sushmitanandani127686-purple.svg)](https://github.com/sushmitanandani127686)

> **IBM SkillsBuild Internship Submission**  
> **Domain:** Applied Machine Learning & Sustainable Agriculture  
> **Developer:** Sushmita Nandani  
> **Repository:** [https://github.com/sushmitanandani127686/AgriSmart-AI](https://github.com/sushmitanandani127686/AgriSmart-AI)

---

## 1. Project Overview & Problem Statement

Agriculture is the backbone of food security and rural livelihoods, yet smallholder farmers frequently suffer significant yield losses and soil degradation due to mismatched crop selection. Traditional farming decisions often rely on trial-and-error, generational habit, or static seasonal calendars that fail to account for specific soil chemical compositions and changing regional rainfall patterns.

**AgriSmart AI** addresses this challenge by providing an intelligent, empirical decision-support system. Using a supervised ensemble machine learning model trained on historical agro-climatic trial data, the system evaluates seven critical soil macronutrient and climatic variables:
- **Soil Chemistry:** Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), and Soil Acidity/Alkalinity ($\text{pH}$).
- **Environmental Factors:** Ambient Temperature ($^\circ\text{C}$), Relative Atmospheric Humidity ($\%$), and Annual/Seasonal Rainfall ($\text{mm}$).

The application recommends the most suitable crop out of **22 agricultural commodities** and pairs each recommendation with practical agronomic profiles (soil guidelines, irrigation advice, growth duration, and management tips) curated from the Food and Agriculture Organization (FAO) and the Indian Council of Agricultural Research (ICAR).

---

## 2. Intended Users & Objectives

### Target Audience
- **Smallholder Farmers & Landholders:** Looking to diversify crop portfolios and maximize harvest success based on soil health test cards.
- **Agricultural Extension Officers & Field Agents:** Providing scientific recommendations to village farming clusters.
- **Agronomy Students & Researchers:** Studying multiclass classification applied to precision agriculture.

### Key Objectives
1. **Precision Matching:** Achieve $>98\%$ classification accuracy across 22 distinct crop classes using genuine supervised learning.
2. **Zero Data Leakage:** Ensure training and preprocessing pipelines are strictly isolated from held-out test evaluation.
3. **Decoupled Architecture:** Maintain clean separation between the data validation layer, training pipeline, prediction engine, and Streamlit user interface.
4. **Actionable Insights:** Deliver transparent confidence percentages, top alternative candidates, and curated crop profiles.
5. **Responsible AI:** Provide clear bounding limits, transparent evaluation metrics, and an agronomic scope disclaimer.

---

## 3. System Architecture & Workflow

```
+-------------------------------------------------------------------------+
|                        AgriSmart AI Architecture                        |
+-------------------------------------------------------------------------+

  [ data/Crop_recommendation.csv ] (2,200 samples, 22 balanced classes)
                 │
                 ▼
  [ src/data_loader.py ] ──► Schema Verification & Range Validation
                 │
                 ▼
  [ train_model.py ] ──────► Stratified 80/20 Train-Test Split (Seed: 42)
        │            │
        │            ├─► Baseline: Decision Tree Classifier
        │            └─► Primary:  Random Forest Classifier (100 Trees)
        ▼
  [ Held-Out Evaluation ] ─► Zero Leakage (Accuracy, Macro Precision/Recall/F1)
        │
        ├─► [ models/crop_rf_model.joblib ]  (Serialized Model Artifact)
        └─► [ models/model_metrics.json ]    (Verified Evaluation Metrics)
                 │
                 ▼
  [ src/predict.py ] ─────► Input Sanitization & Agronomic Profiles (FAO/ICAR)
                 │
                 ▼
  [ app.py (Streamlit) ] ─► Interactive Web UI (Emerald/Sage Theme)
                 │
                 ├─► 🌾 Real-time Validated Prediction & Top Candidates
                 ├─► 📊 Measured Model Performance & Confusion Matrix (Plotly)
                 ├─► 🌱 Dataset Distribution & Feature Importance (Plotly)
                 ├─► 🕒 Active Session Prediction History (CSV Export)
                 └─► ℹ️ Methodology & Internship Documentation
```

---

## 4. Technology Stack

- **Programming Language:** Python 3.9+
- **Data Manipulation & Analysis:** pandas (`2.2.3`), NumPy (`1.26.4`)
- **Machine Learning & Modeling:** scikit-learn (`1.5.2`)
- **Model Serialization:** joblib (`1.5.3`)
- **Interactive Web Interface:** Streamlit (`1.39.1`)
- **Interactive Visualizations:** Plotly (`7.1.0`)
- **Testing & Quality Assurance:** pytest (`8.4.2`)
- **Version Control:** Git & GitHub

---

## 5. Dataset Attribution & Provenance

- **Dataset Name:** Crop Recommendation Dataset (Indian Council of Agricultural Research & Open Agriculture trials)
- **Sample Count:** 2,200 complete records
- **Class Balance:** Perfectly balanced across 22 crops (100 samples per crop class)
- **Target Classes (22):** Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.
- **License:** Public Domain / Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Limitations:** Data reflects static, healthy crop conditions without modeling transient insect pest infestations, severe frost events, or soil microbiome depletion.

---

## 6. Installation & Setup (macOS / Linux)

### Prerequisites
- Python 3.9 or higher
- Git installed

### Step-by-Step Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sushmitanandani127686/AgriSmart-AI.git
   cd AgriSmart-AI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade pip and install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. Model Training & Evaluation

The training script automatically performs data loading, schema checks, train/test splitting, baseline benchmarking, and model serialization:

```bash
python train_model.py
```

### Measured Evaluation Results (Verified on 440 Held-Out Test Samples)

The models were evaluated strictly on an untouched 20% held-out test split ($n=440$) using stratified sampling:

| Evaluation Metric | Baseline (Decision Tree) | **Primary (Random Forest)** |
| :--- | :---: | :---: |
| **Accuracy** | 97.95% | **99.55%** |
| **Macro Precision** | 98.06% | **99.57%** |
| **Macro Recall** | 97.95% | **99.55%** |
| **Macro F1-Score** | 97.94% | **99.55%** |
| **Data Leakage Risk** | Zero (Held-out Split) | **Zero (Held-out Split)** |

### Gini Feature Importances (Random Forest)
1. **Rainfall:** 23.02%
2. **Relative Humidity:** 22.42%
3. **Potassium ($K$):** 17.54%
4. **Phosphorus ($P$):** 15.08%
5. **Nitrogen ($N$):** 9.64%
6. **Ambient Temperature:** 7.24%
7. **Soil pH:** 5.06%

---

## 8. Launching the Web Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

Once started, the application will be accessible at:
```text
Local URL: http://localhost:8501
Network URL: http://<your-local-ip>:8501
```

---

## 9. Automated Testing

To run the complete automated test suite covering schema validation, edge case boundaries, missing features, and inference:

```bash
pytest -v tests/
```

**Expected output:**
```text
tests/test_prediction.py::test_dataset_loading_and_schema PASSED
tests/test_prediction.py::test_dataset_missing_file_raises PASSED
tests/test_prediction.py::test_data_summary_integrity PASSED
tests/test_prediction.py::test_model_artifacts_exist PASSED
tests/test_prediction.py::test_prediction_engine_valid_inference PASSED
tests/test_prediction.py::test_prediction_engine_alternative_crop PASSED
tests/test_prediction.py::test_prediction_boundary_validation_fails PASSED
tests/test_prediction.py::test_prediction_missing_features_fails PASSED
tests/test_prediction.py::test_all_22_crop_profiles_exist PASSED
============================== 9 passed in 1.14s ===============================
```

---

## 10. Example Inputs & Output

### Scenario: Monsoonal Paddy Field
- **Inputs:**
  - Nitrogen ($N$): $90\text{ kg/ha}$
  - Phosphorus ($P$): $42\text{ kg/ha}$
  - Potassium ($K$): $43\text{ kg/ha}$
  - Temperature: $20.8^\circ\text{C}$
  - Humidity: $82.0\%$
  - Soil pH: $6.5$
  - Rainfall: $202.9\text{ mm}$
- **Predicted Crop:** **Rice (*Oryza sativa*)**
- **Model Confidence:** $90.0\%$
- **Agronomic Guidance:** High water requirement ($1000 - 2500\text{ mm}$). Maintain puddling in fields to conserve moisture and suppress weeds.

---

## 11. Limitations & Responsible-Use Disclaimer

> **Responsible-Use Notice:**  
> AgriSmart AI is an educational and advisory decision-support tool. Crop suitability predictions are based on statistical pattern matching of historical chemical and climatic boundaries. The model **does not**:
> 1. Guarantee agricultural harvest yields or financial profits.
> 2. Replace certified laboratory soil testing or on-site agricultural extension consultations.
> 3. Provide chemical pesticide prescriptions or active disease diagnosis.
> 4. Measure real-time IoT soil sensor or satellite weather feeds automatically.

---

## 12. GitHub Repository Setup Commands

To push this repository to your GitHub account (`sushmitanandani127686`):

```bash
# 1. Check current git status
git status

# 2. Add all project files
git add .

# 3. Commit the verified code
git commit -m "feat: complete AgriSmart AI crop recommendation system for IBM SkillsBuild internship"

# 4. Set the main branch
git branch -M main

# 5. Add your GitHub remote repository
git remote add origin https://github.com/sushmitanandani127686/AgriSmart-AI.git

# 6. Push to GitHub
git push -u origin main
```

---

## 13. License

Distributed under the MIT License. See `LICENSE` for more information.
