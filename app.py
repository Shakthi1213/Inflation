import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Inflation Prediction Tool",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        border-bottom: 3px solid #2E74B5;
        padding-bottom: 0.8rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1F4E79;
        border-left: 4px solid #2E74B5;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv("regression_final.csv")
    df["Period"] = pd.to_datetime(df["Period"])
    df = df.sort_values("Period").reset_index(drop=True)
    if "inflation_lag1" not in df.columns:
        df["inflation_lag1"] = df["Inflation_Rate_YoY"].shift(1)
        df = df.dropna().reset_index(drop=True)
    return df


df = load_data()
IVS = [
    "WPI_yoy",
    "IIP_yoy",
    "Imports_yoy",
    "MarketBorrowing_yoy",
    "CPI_food_lag1_yoy",
    "inflation_lag1",
]
DV = "Inflation_Rate_YoY"


@st.cache_data
def fit_model(data):
    x = sm.add_constant(data[IVS])
    y = data[DV]
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})


model = fit_model(df)

st.markdown('<div class="main-title">Inflation Prediction Tool</div>', unsafe_allow_html=True)
st.markdown("Enter macroeconomic values to predict India's CPI inflation rate.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Input Variables**")
    wpi = st.number_input("WPI YoY %", min_value=-20.0, max_value=30.0, value=3.5, step=0.1)
    iip = st.number_input("IIP YoY %", min_value=-30.0, max_value=50.0, value=5.0, step=0.1)
with col2:
    st.markdown("**Input Variables (cont.)**")
    imports = st.number_input("Imports YoY %", min_value=-60.0, max_value=120.0, value=8.0, step=0.5)
    mkt_borrow = st.number_input(
        "Market Borrowing YoY %",
        min_value=-100.0,
        max_value=150.0,
        value=10.0,
        step=1.0,
    )
with col3:
    st.markdown("**Lagged Variables**")
    food_lag = st.number_input(
        "CPI Food Lag1 YoY %",
        min_value=-5.0,
        max_value=20.0,
        value=5.5,
        step=0.1,
    )
    inf_lag = st.number_input("Inflation Lag1 %", min_value=-2.0, max_value=12.0, value=4.5, step=0.1)

st.markdown("")
col_btn = st.columns([1, 3, 1])
with col_btn[0]:
    predict_btn = st.button("Predict", type="primary", use_container_width=True)

if predict_btn:
    input_vals = np.array([1.0, wpi, iip, imports, mkt_borrow, food_lag, inf_lag])
    prediction = float(model.params @ input_vals)
    ci_lo_pred = prediction - 1.96 * np.sqrt(model.scale)
    ci_hi_pred = prediction + 1.96 * np.sqrt(model.scale)

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Predicted CPI Inflation", f"{prediction:.2f}%")
    col_r2.metric("95% CI Lower", f"{ci_lo_pred:.2f}%")
    col_r3.metric("95% CI Upper", f"{ci_hi_pred:.2f}%")

    if prediction <= 4.0:
        st.success(f"Predicted inflation of {prediction:.2f}% is within the RBI target.")
    elif prediction <= 6.0:
        st.warning(f"Predicted inflation of {prediction:.2f}% is above target but within the tolerance band.")
    else:
        st.error(f"Predicted inflation of {prediction:.2f}% exceeds the RBI upper tolerance.")

    st.markdown('<div class="section-header">Variable Contributions to Prediction</div>', unsafe_allow_html=True)
    inputs_no_const = [wpi, iip, imports, mkt_borrow, food_lag, inf_lag]
    contribs = [model.params[var] * value for var, value in zip(IVS, inputs_no_const)]

    contrib_df = pd.DataFrame(
        {
            "Variable": IVS,
            "Input Value": [round(x, 2) for x in inputs_no_const],
            "Coefficient": [round(model.params[var], 4) for var in IVS],
            "Contribution (%)": [round(val, 4) for val in contribs],
        }
    )
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#0F6E56" if c > 0 else "#E24B4A" for c in contribs]
    bars = ax.barh(IVS, contribs, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Contribution to Predicted Inflation (%)", fontsize=11, fontweight="bold")
    ax.set_title("Impact of Each Variable on Inflation Prediction", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    for bar, val in zip(bars, contribs):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("---")
st.markdown('<div class="section-header">About This Tool</div>', unsafe_allow_html=True)

info1, info2 = st.columns(2)
with info1:
    st.markdown(
        f"""
    **Model Details:**
    - **Type:** OLS Regression with HAC Standard Errors
    - **Estimation:** Newey-West (3 lags)
    - **R-squared:** {model.rsquared:.4f}
    - **Adjusted R-squared:** {model.rsquared_adj:.4f}
    - **Observations:** {len(df)} months
    """
    )

with info2:
    st.markdown(
        """
    **Variables Included:**
    - WPI growth
    - IIP growth
    - Import growth
    - Government borrowing
    - Lagged food inflation
    - Lagged inflation
    """
    )

st.markdown("")
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#888;font-size:0.85rem;margin-top:1rem'>
    <b>India Inflation Prediction Tool</b> | OLS Regression with HAC Standard Errors |
    Data: RBI DBIE, MOSPI, OEA | Oct 2018 - Jun 2025
    </div>
    """,
    unsafe_allow_html=True,
)
