import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. 页面设置
# ==========================================
st.set_page_config(
    page_title="ACS MACCE Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# 2. 自定义页面样式
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
# 3. 标题区
# ==========================================
st.markdown(
    """
    <div class="title-box">
        <h1>ACS住院期间MACCE风险预测工具</h1>
        <p>基于 CatBoost 模型的机器学习预测系统</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 4. 路径与变量
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
# 5. 加载模型
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
    st.error("模型加载失败，请检查模型文件路径或 joblib 文件结构。")
    st.exception(e)
    st.stop()

# ==========================================
# 6. 侧边栏说明
# ==========================================
with st.sidebar:
    st.header("模型信息")
    st.write("**特征数量：** 11")
    st.write("**算法：** CatBoost")
    st.write("**预测结局：** 住院期间 MACCE")
    st.write("**输出结果：** MACCE 发生概率")

    st.divider()

    st.header("风险分层规则")
    st.write("低危：预测概率 < 10%")
    st.write("中危：10% ≤ 预测概率 < 30%")
    st.write("高危：预测概率 ≥ 30%")

    st.divider()

    st.caption(
        "注：风险分层阈值目前为展示用规则。正式论文中建议根据训练集、验证集或临床决策需求确定。"
    )

# ==========================================
# 7. 输入区
# ==========================================
st.subheader("请输入患者临床指标")

# ------------------------------
# 7.1 基本信息与连续变量
# ------------------------------
st.markdown("### 基本信息与连续变量")

basic_col1, basic_col2, basic_col3 = st.columns(3)

with basic_col1:
    Age = st.number_input(
        "年龄 Age",
        min_value=18,
        max_value=120,
        value=65,
        step=1
    )

with basic_col2:
    LVEF = st.number_input(
        "左室射血分数 LVEF (%)",
        min_value=1.0,
        max_value=100.0,
        value=55.0,
        step=1.0,
        format="%.2f"
    )

with basic_col3:
    FBG = st.number_input(
        "空腹血糖 FBG",
        min_value=0.0,
        max_value=50.0,
        value=5.6,
        step=0.1,
        format="%.1f"
    )

basic_col4, basic_col5, basic_col6 = st.columns(3)

with basic_col4:
    aSI = st.number_input(
        "年龄休克指数 aSI",
        min_value=0.0,
        max_value=5.0,
        value=0.70,
        step=0.01,
        format="%.2f"
    )

with basic_col5:
    AIP = st.number_input(
        "动脉粥样硬化指数 AIP",
        min_value=-5.0,
        max_value=5.0,
        value=0.20,
        step=0.01,
        format="%.2f"
    )

with basic_col6:
    eGFR = st.selectbox(
        "eGFR 分组",
        options=["<30", "30-60", "60-90", ">90"],
        index=2
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# 7.2 临床状态与治疗变量
# ------------------------------
st.markdown("### 临床状态与治疗变量")

clinical_col1, clinical_col2, clinical_col3 = st.columns(3)

with clinical_col1:
    AHF = st.selectbox(
        "急性心力衰竭 AHF",
        options=["No", "Yes"]
    )

with clinical_col2:
    Killip_class = st.selectbox(
        "Killip class",
        options=["I", "II", "III", "IV"]
    )

with clinical_col3:
    Aldosterone = st.selectbox(
        "住院期间使用醛固酮受体拮抗剂",
        options=["No", "Yes"]
    )

clinical_col4, clinical_col5, clinical_col6 = st.columns(3)

with clinical_col4:
    ST_deviation = st.selectbox(
        "ST段偏移",
        options=["No", "Yes"]
    )

with clinical_col5:
    Shock = st.selectbox(
        "心源性休克 Cardiogenic shock",
        options=["No", "Yes"]
    )

with clinical_col6:
    st.empty()
# ==========================================
# 8. 生成输入 DataFrame
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
# 9. 预测函数
# ==========================================
def predict_macce_probability(input_df, pipeline, train_cols):
    X_enc = pd.get_dummies(input_df[MODEL5_FEATURES], drop_first=True)
    X_enc = X_enc.reindex(columns=train_cols, fill_value=0)
    pred_prob = pipeline.predict_proba(X_enc)[:, 1][0]
    return pred_prob, X_enc


def get_risk_group(pred_prob):
    if pred_prob < 0.10:
        return {
            "group": "低危",
            "class": "risk-low",
            "advice": "预测风险较低，可结合临床情况进行常规监测与管理。"
        }
    elif pred_prob < 0.30:
        return {
            "group": "中危",
            "class": "risk-medium",
            "advice": "预测风险中等，建议加强病情观察，并结合其他临床指标综合评估。"
        }
    else:
        return {
            "group": "高危",
            "class": "risk-high",
            "advice": "预测风险较高，建议重点监测，并根据临床情况考虑更积极的干预策略。"
        }


# ==========================================
# 10. 预测按钮与结果展示
# ==========================================
st.divider()

predict_button = st.button("开始预测 MACCE 风险", use_container_width=True)

if predict_button:
    try:
        pred_prob, X_enc = predict_macce_probability(input_df, pipeline, train_cols)
        risk_info = get_risk_group(pred_prob)

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-title">住院期间 MACCE 预测概率</p>
                    <div class="probability-text">{pred_prob * 100:.2f}%</div>
                    <p class="result-desc">
                        模型估计的住院期间<br>MACCE 发生概率。
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-title">风险分层</p>
                    <div class="{risk_info["class"]}">{risk_info["group"]}</div>
                    <p class="result-desc">
                        {risk_info["advice"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )



    except Exception as e:
        st.error("预测失败，请检查输入数据或模型文件。")
        st.exception(e)
# ==========================================
# 11. 页面底部声明
# ==========================================
st.markdown(
    """
    <div class="footer-box">
        <b>说明：</b><br>
        本工具基于机器学习模型对 ACS 患者住院期间 MACCE 发生风险进行预测，
        仅用于科研展示和风险辅助评估，不能替代临床医生的专业判断。
        实际应用时应结合患者病情、实验室检查、影像学结果及医生经验进行综合决策。
    </div>
    """,
    unsafe_allow_html=True
)