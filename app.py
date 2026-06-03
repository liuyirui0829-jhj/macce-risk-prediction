
import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. Page configuration
# ==========================================
st.set_page_config(
    page_title="ACS MACCE Risk Prediction",
    page_icon="❤️",
    layout="wide"
)
# ==========================================
# 2. Custom CSS
# ==========================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .title-box {
        background: linear-gradient(90deg, #8B0000, #B22222);
        padding: 28px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .title-box h1 {
        font-size: 34px;
        margin-bottom: 8px;
    }
    .title-box p {
        font-size: 17px;
        margin-bottom: 0px;
    }
    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    .result-card {
        background-color: white;
        padding: 28px 30px;
        border-radius: 18px;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
        margin-top: 15px;
        text-align: center;
        height: 300px;
        display: grid;
        grid-template-rows: 60px 110px 80px;
        align-items: center;
    }
    .result-title {
        font-size: 24px;
        color: #4b5563;
        font-weight: 700;
        margin: 0;
    }
    .probability-text {
        font-size: 64px;
        font-weight: 800;
        color: #111827;
        line-height: 1;
        margin: 0;
    }
    .risk-low {
        color: #15803d;
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }
    .risk-medium {
        color: #d97706;
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }
    .risk-high {
        color: #b91c1c;
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }
    .result-desc {
        font-size: 18px;
        color: #6b7280;
        line-height: 1.6;
        font-weight: 600;
        margin: 0;
    }

    .small-note {
        font-size: 13px;
        color: #6b7280;
        line-height: 1.6;
    }
    .definition-box {
        background-color: #f9fafb;
        border-left: 5px solid #991b1b;
        border-right: 5px solid #991b1b;
        padding: 18px 35px;
        border-radius: 12px;
        margin-bottom: 26px;
        font-size: 16px;
        line-height: 1.7;
        color: #374151;
    }
        .info-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 18px;
        margin-bottom: 28px;
    }

    .info-card {
        background-color: white;
        padding: 20px 24px;
        border-radius: 14px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
        font-size: 16px;
        line-height: 1.8;
        color: #374151;
    }

    .info-card h3 {
        margin-top: 0;
        margin-bottom: 12px;
        color: #991b1b;
        font-size: 20px;
    }
    .footer-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 16px;
        border-radius: 10px;
        margin-top: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==========================================
# 3. Header
# ==========================================
st.markdown(
    """
    <div class="title-box">
        <h1>In-hospital MACCE Risk Prediction Tool for ACS Patients</h1>
        <p>A machine learning prediction system based on the CatBoost model</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="definition-box">
        <b>Definition of MACCE:</b>
        In-hospital major adverse cardiac and cerebrovascular events (MACCE) 
        were defined as a composite of all-cause death, recurrent myocardial infarction, 
        acute thrombosis, acute heart failure, and stroke.
    </div>
    """,
    unsafe_allow_html=True
)
# ==========================================
# Model information and risk stratification
# ==========================================
info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.container(border=True):
        st.markdown("### Model Information")
        st.markdown("**Number of features:** 11")
        st.markdown("**Algorithm:** CatBoost")
        st.markdown("**Predicted outcome:** In-hospital MACCE")
        st.markdown("**Output:** Probability of MACCE")

with info_col2:
    with st.container(border=True):
        st.markdown("### Risk Stratification")
        st.markdown("**Low risk:** predicted probability < 2.00%")
        st.markdown("**Intermediate risk:** 2.00% ≤ predicted probability < 8.00%")
        st.markdown("**High risk:** predicted probability ≥ 8.00%")
        st.caption(
            "The risk stratification thresholds were determined based on the distribution "
            "of predicted probabilities in the training set and are applied for risk presentation "
            "in this web application."
        )
# ==========================================
# 4. Paths and variables
# ==========================================
MODEL_PATH = "saved_models/Model_5_11_Features_-_CatBoost.joblib"
MODEL5_FEATURES = [
    "AHF",
    "LVEF",
    "Killip class",
    "Age",
    "aSI",
    "eGFR",
    "In hospital aldosterone receptor antagonists",
    "FBG",
    "ST segment deviation",
    "AIP",
    "Cardiogenic shock"
]


# ==========================================
# 5. Load model
# ==========================================
@st.cache_resource
def load_model():
    obj = joblib.load(MODEL_PATH)
    pipeline = obj["pipeline"]
    train_cols = obj["columns"]
    return pipeline, train_cols


try:
    pipeline, train_cols = load_model()
except Exception as e:
    st.error("Failed to load the model. Please check the model file path or the joblib file structure.")
    st.exception(e)
    st.stop()

# ==========================================
# 7. Input section
# ==========================================
st.subheader("Enter Patient Clinical Variables")
# ------------------------------
# 7.1 Basic information and continuous variables
# ------------------------------
st.markdown("### Basic Information and Continuous Variables")
basic_col1, basic_col2, basic_col3 = st.columns(3)
with basic_col1:
    Age = st.number_input(
        "Age, years",
        min_value=18,
        max_value=120,
        value=65,
        step=1,
        help="Patient age in years."
    )
with basic_col2:
    LVEF = st.number_input(
        "Left ventricular ejection fraction, %",
        min_value=1.0,
        max_value=100.0,
        value=55.0,
        step=1.0,
        format="%.2f",
        help="Left ventricular ejection fraction, expressed as a percentage."
    )
with basic_col3:
    FBG = st.number_input(
        "Fasting blood glucose, mmol/L",
        min_value=0.0,
        max_value=50.0,
        value=5.6,
        step=0.1,
        format="%.1f",
        help="Fasting blood glucose, measured in mmol/L."
    )
basic_col4, basic_col5, basic_col6 = st.columns(3)
with basic_col4:
    aSI = st.number_input(
        "Age-adjusted shock index",
        min_value=0.0,
        max_value=5.0,
        value=0.70,
        step=0.01,
        format="%.2f",
        help="Age-adjusted shock index, an index-type variable."
    )
with basic_col5:
    AIP = st.number_input(
        "Atherogenic index of plasma",
        min_value=-5.0,
        max_value=5.0,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="Atherogenic index of plasma, usually calculated from lipid parameters and treated as a unitless index."
    )
with basic_col6:
    eGFR = st.selectbox(
        "eGFR group, mL/min/1.73 m²",
        options=["<30", "30-60", "60-90", ">90"],
        index=2,
        help="Estimated glomerular filtration rate group, measured in mL/min/1.73 m²."
    )
st.markdown("<br>", unsafe_allow_html=True)
# ------------------------------
# 7.2 Clinical status and treatment variables
# ------------------------------
st.markdown("### Clinical Status and Treatment Variables")
clinical_col1, clinical_col2, clinical_col3 = st.columns(3)
with clinical_col1:
    AHF = st.selectbox(
        "Acute heart failure",
        options=["No", "Yes"],
        help="Whether acute heart failure was present."
    )
with clinical_col2:
    Killip_class = st.selectbox(
        "Killip class",
        options=["I", "II", "III", "IV"],
        help="Killip classification, ranging from I to IV."
    )
with clinical_col3:
    Aldosterone = st.selectbox(
        "In-hospital MRA use",
        options=["No", "Yes"],
        help="Whether mineralocorticoid receptor antagonists, also known as aldosterone receptor antagonists, were used during hospitalization."
    )
clinical_col4, clinical_col5, clinical_col6 = st.columns(3)
with clinical_col4:
    ST_deviation = st.selectbox(
        "ST-segment deviation",
        options=["No", "Yes"],
        help="Whether ST-segment elevation or depression was present."
    )
with clinical_col5:
    Shock = st.selectbox(
        "Cardiogenic shock",
        options=["No", "Yes"],
        help="Whether cardiogenic shock was present."
    )
with clinical_col6:
    st.empty()
# ==========================================
# 8. Generate input DataFrame
# ==========================================
input_df = pd.DataFrame({
    "AHF": [AHF],
    "LVEF": [LVEF],
    "Killip class": [Killip_class],
    "Age": [Age],
    "aSI": [aSI],
    "eGFR": [eGFR],
    "In hospital aldosterone receptor antagonists": [Aldosterone],
    "FBG": [FBG],
    "ST segment deviation": [ST_deviation],
    "AIP": [AIP],
    "Cardiogenic shock": [Shock]
})


# ==========================================
# 9. Prediction functions
# ==========================================
def predict_macce_probability(input_df, pipeline, train_cols):
    """
    Manually encode one patient's input to avoid incorrect one-hot encoding
    when predicting a single patient.
    """

    row = input_df.iloc[0]

    # Use 0.0 instead of 0 so the DataFrame can store decimal values
    X_enc = pd.DataFrame(0.0, index=[0], columns=train_cols)

    # Continuous variables
    continuous_vars = ["LVEF", "Age", "aSI", "FBG", "AIP"]

    for var in continuous_vars:
        if var in X_enc.columns:
            X_enc.loc[0, var] = float(row[var])

    # Binary categorical variables
    binary_vars = [
        "AHF",
        "In hospital aldosterone receptor antagonists",
        "ST segment deviation",
        "Cardiogenic shock"
    ]

    for var in binary_vars:
        col_yes = f"{var}_Yes"
        if col_yes in X_enc.columns:
            X_enc.loc[0, col_yes] = 1.0 if row[var] == "Yes" else 0.0

    # Killip class
    killip_value = row["Killip class"]
    killip_col = f"Killip class_{killip_value}"

    if killip_col in X_enc.columns:
        X_enc.loc[0, killip_col] = 1.0

    # eGFR group
    egfr_value = row["eGFR"]
    egfr_col = f"eGFR_{egfr_value}"

    if egfr_col in X_enc.columns:
        X_enc.loc[0, egfr_col] = 1.0

    pred_prob = pipeline.predict_proba(X_enc)[:, 1][0]

    return pred_prob, X_enc

def get_risk_group(pred_prob):
    if pred_prob < 0.02:
        return {
            "group": "Low Risk",
            "class": "risk-low",
            "advice": "The predicted risk is relatively low. Routine monitoring and management may be considered according to the clinical context."
        }
    elif pred_prob < 0.08:
        return {
            "group": "Intermediate Risk",
            "class": "risk-medium",
            "advice": "The predicted risk is intermediate. Closer observation and comprehensive clinical assessment are recommended."
        }
    else:
        return {
            "group": "High Risk",
            "class": "risk-high",
            "advice": "The predicted risk is high. Intensive monitoring and more proactive management strategies should be considered according to the clinical context."
        }


# ==========================================
# 10. Prediction button and result display
# ==========================================
st.divider()
predict_button = st.button("Predict MACCE Risk", use_container_width=True)
if predict_button:
    try:
        pred_prob, X_enc = predict_macce_probability(input_df, pipeline, train_cols)
        risk_info = get_risk_group(pred_prob)
        result_col1, result_col2 = st.columns([1, 1])
        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-title">Predicted Probability of In-hospital MACCE</p>
                    <div class="probability-text">{pred_prob * 100:.2f}%</div>
                    <p class="result-desc">
                        Model-estimated probability of<br>in-hospital MACCE.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-title">Risk Stratification</p>
                    <div class="{risk_info["class"]}">{risk_info["group"]}</div>
                    <p class="result-desc">
                        {risk_info["advice"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error("Prediction failed. Please check the input data or the model file.")
        st.exception(e)
# ==========================================
# 11. Footer
# ==========================================
st.markdown(
    """
    <div class="footer-box">
        <b>Disclaimer:</b><br>
        This tool uses a machine learning model to estimate the risk of in-hospital MACCE 
        among patients with acute coronary syndrome. It is intended for research demonstration 
        and risk-assessment support only and should not replace professional clinical judgment. 
        In real-world practice, decisions should be made comprehensively based on the patient's 
        clinical condition, laboratory findings, imaging results, and physician expertise.
    </div>
    """,
    unsafe_allow_html=True
)