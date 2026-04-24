import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inflation Prediction Tool",
    page_icon="📈",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700; color: #1F4E79;
        border-bottom: 3px solid #2E74B5; padding-bottom: 0.8rem; margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #1F4E79;
        border-left: 4px solid #2E74B5; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("regression_final.csv")
    df['Period'] = pd.to_datetime(df['Period'])
    df = df.sort_values('Period').reset_index(drop=True)
    if 'inflation_lag1' not in df.columns:
        df['inflation_lag1'] = df['Inflation_Rate_YoY'].shift(1)
        df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

IVs = ['WPI_yoy', 'IIP_yoy', 'Imports_yoy', 'MarketBorrowing_yoy',
       'CPI_food_lag1_yoy', 'inflation_lag1']
DV = 'Inflation_Rate_YoY'

# ── FIT MODEL ─────────────────────────────────────────────────────────────────
@st.cache_data
def fit_model(data):
    X = sm.add_constant(data[IVs])
    y = data[DV]
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return model

model = fit_model(df)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION TOOL PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">📈 Inflation Prediction Tool</div>', unsafe_allow_html=True)
st.markdown("**Enter macroeconomic values to predict India's CPI inflation rate**")
st.markdown("")

# Input section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Input Variables**")
    wpi = st.number_input(
        "WPI YoY %",
        min_value=-20.0, max_value=30.0, value=3.5, step=0.1,
        help="Wholesale Price Index Year-on-Year growth"
    )
    iip = st.number_input(
        "IIP YoY %",
        min_value=-30.0, max_value=50.0, value=5.0, step=0.1,
        help="Industrial Production Year-on-Year growth"
    )

with col2:
    st.markdown("**Input Variables (cont.)**")
    imports = st.number_input(
        "Imports YoY %",
        min_value=-60.0, max_value=120.0, value=8.0, step=0.5,
        help="Import Value Year-on-Year growth"
    )
    mkt_borrow = st.number_input(
        "Market Borrowing YoY %",
        min_value=-100.0, max_value=150.0, value=10.0, step=1.0,
        help="Government Market Borrowing YoY growth"
    )

with col3:
    st.markdown("**Lagged Variables**")
    food_lag = st.number_input(
        "CPI Food Lag1 YoY %",
        min_value=-5.0, max_value=20.0, value=5.5, step=0.1,
        help="Previous month food inflation"
    )
    inf_lag = st.number_input(
        "Inflation Lag1 %",
        min_value=-2.0, max_value=12.0, value=4.5, step=0.1,
        help="Previous month CPI inflation"
    )

# Prediction button
st.markdown("")
col_btn = st.columns([1, 3, 1])
with col_btn[0]:
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# ── PREDICTION OUTPUT ─────────────────────────────────────────────────────────
if predict_btn:
    input_vals = np.array([[1.0, wpi, iip, imports, mkt_borrow, food_lag, inf_lag]])
    prediction = model.params @ input_vals[0]
    ci_lo_pred = prediction - 1.96 * np.sqrt(model.scale)
    ci_hi_pred = prediction + 1.96 * np.sqrt(model.scale)

    st.markdown("---")
    
    # Results metrics
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Predicted CPI Inflation", f"{prediction:.2f}%", delta=f"CI: [{ci_lo_pred:.2f}%, {ci_hi_pred:.2f}%]")
    col_r2.metric("RBI Target", "4.0%", delta_color="inverse")
    col_r3.metric("Upper Band", "6.0%", delta_color="inverse")

    # Status message
    st.markdown("")
    if prediction <= 4.0:
        st.success(f"✅ **Within RBI Target** — Predicted inflation of {prediction:.2f}% is within the RBI target of 4%", icon="✓")
    elif prediction <= 6.0:
        st.warning(f"⚠️ **Above Target** — Predicted inflation of {prediction:.2f}% is within tolerance band (4–6%)", icon="⚠️")
    else:
        st.error(f"❌ **Above Tolerance** — Predicted inflation of {prediction:.2f}% exceeds upper tolerance of 6%", icon="✕")

    # Variable contributions
    st.markdown('<div class="section-header">Variable Contributions to Prediction</div>', unsafe_allow_html=True)
    inputs_no_const = [wpi, iip, imports, mkt_borrow, food_lag, inf_lag]
    contribs = [model.params[v] * inp for v, inp in zip(IVs, inputs_no_const)]
    
    contrib_df = pd.DataFrame({
        'Variable': IVs,
        'Input Value': [round(x, 2) for x in inputs_no_const],
        'Coefficient': [round(model.params[v], 4) for v in IVs],
        'Contribution (%)': [round(c, 4) for c in contribs]
    })
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ['#0F6E56' if c > 0 else '#E24B4A' for c in contribs]
    bars = ax.barh(IVs, contribs, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Contribution to Predicted Inflation (%)', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Each Variable on Inflation Prediction', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    sns.despine(ax=ax)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, contribs)):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── INFORMATION SECTION ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">ℹ️ About This Tool</div>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown(f"""
    **Model Details:**
    - **Type:** OLS Regression with HAC Standard Errors
    - **Estimation:** Newey-West (3 lags)
    - **R² (In-sample):** {model.rsquared:.4f}
    - **Adjusted R²:** {model.rsquared_adj:.4f}
    - **Observations:** {len(df)} months (Oct 2018 – Jun 2025)
    """)

with col_info2:
    st.markdown("""
    **Variables Included:**
    - WPI growth (cost-push channel)
    - IIP growth (demand-pull channel)
    - Import growth (supply augmentation)
    - Government borrowing (monetary/fiscal)
    - Lagged food inflation (persistence)
    - Lagged inflation (AR effects)
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;margin-top:1rem'>"
    "<b>India Inflation Prediction Tool</b> | OLS Regression with HAC Standard Errors | "
    "Data: RBI DBIE, MOSPI, OEA | Oct 2018 – Jun 2025<br>"
    "<i>Macroeconomic Determinants of India's CPI Inflation</i>"
    "</div>",
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inflation Prediction Tool",
    page_icon="📈",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700; color: #1F4E79;
        border-bottom: 3px solid #2E74B5; padding-bottom: 0.8rem; margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #1F4E79;
        border-left: 4px solid #2E74B5; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("regression_final.csv")
    df['Period'] = pd.to_datetime(df['Period'])
    df = df.sort_values('Period').reset_index(drop=True)
    if 'inflation_lag1' not in df.columns:
        df['inflation_lag1'] = df['Inflation_Rate_YoY'].shift(1)
        df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

IVs = ['WPI_yoy', 'IIP_yoy', 'Imports_yoy', 'MarketBorrowing_yoy',
       'CPI_food_lag1_yoy', 'inflation_lag1']
DV = 'Inflation_Rate_YoY'

# ── FIT MODEL ─────────────────────────────────────────────────────────────────
@st.cache_data
def fit_model(data):
    X = sm.add_constant(data[IVs])
    y = data[DV]
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return model

model = fit_model(df)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION TOOL PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">📈 Inflation Prediction Tool</div>', unsafe_allow_html=True)
st.markdown("**Enter macroeconomic values to predict India's CPI inflation rate**")
st.markdown("")

# Input section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Input Variables**")
    wpi = st.number_input(
        "WPI YoY %",
        min_value=-20.0, max_value=30.0, value=3.5, step=0.1,
        help="Wholesale Price Index Year-on-Year growth"
    )
    iip = st.number_input(
        "IIP YoY %",
        min_value=-30.0, max_value=50.0, value=5.0, step=0.1,
        help="Industrial Production Year-on-Year growth"
    )

with col2:
    st.markdown("**Input Variables (cont.)**")
    imports = st.number_input(
        "Imports YoY %",
        min_value=-60.0, max_value=120.0, value=8.0, step=0.5,
        help="Import Value Year-on-Year growth"
    )
    mkt_borrow = st.number_input(
        "Market Borrowing YoY %",
        min_value=-100.0, max_value=150.0, value=10.0, step=1.0,
        help="Government Market Borrowing YoY growth"
    )

with col3:
    st.markdown("**Lagged Variables**")
    food_lag = st.number_input(
        "CPI Food Lag1 YoY %",
        min_value=-5.0, max_value=20.0, value=5.5, step=0.1,
        help="Previous month food inflation"
    )
    inf_lag = st.number_input(
        "Inflation Lag1 %",
        min_value=-2.0, max_value=12.0, value=4.5, step=0.1,
        help="Previous month CPI inflation"
    )

# Prediction button
st.markdown("")
col_btn = st.columns([1, 3, 1])
with col_btn[0]:
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# ── PREDICTION OUTPUT ─────────────────────────────────────────────────────────
if predict_btn:
    input_vals = np.array([[1.0, wpi, iip, imports, mkt_borrow, food_lag, inf_lag]])
    prediction = model.params @ input_vals[0]
    ci_lo_pred = prediction - 1.96 * np.sqrt(model.scale)
    ci_hi_pred = prediction + 1.96 * np.sqrt(model.scale)

    st.markdown("---")
    
    # Results metrics
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Predicted CPI Inflation", f"{prediction:.2f}%", delta=f"CI: [{ci_lo_pred:.2f}%, {ci_hi_pred:.2f}%]")
    col_r2.metric("RBI Target", "4.0%", delta_color="inverse")
    col_r3.metric("Upper Band", "6.0%", delta_color="inverse")

    # Status message
    st.markdown("")
    if prediction <= 4.0:
        st.success(f"✅ **Within RBI Target** — Predicted inflation of {prediction:.2f}% is within the RBI target of 4%", icon="✓")
    elif prediction <= 6.0:
        st.warning(f"⚠️ **Above Target** — Predicted inflation of {prediction:.2f}% is within tolerance band (4–6%)", icon="⚠️")
    else:
        st.error(f"❌ **Above Tolerance** — Predicted inflation of {prediction:.2f}% exceeds upper tolerance of 6%", icon="✕")

    # Variable contributions
    st.markdown('<div class="section-header">Variable Contributions to Prediction</div>', unsafe_allow_html=True)
    inputs_no_const = [wpi, iip, imports, mkt_borrow, food_lag, inf_lag]
    contribs = [model.params[v] * inp for v, inp in zip(IVs, inputs_no_const)]
    
    contrib_df = pd.DataFrame({
        'Variable': IVs,
        'Input Value': [round(x, 2) for x in inputs_no_const],
        'Coefficient': [round(model.params[v], 4) for v in IVs],
        'Contribution (%)': [round(c, 4) for c in contribs]
    })
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ['#0F6E56' if c > 0 else '#E24B4A' for c in contribs]
    bars = ax.barh(IVs, contribs, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Contribution to Predicted Inflation (%)', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Each Variable on Inflation Prediction', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    sns.despine(ax=ax)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, contribs)):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── INFORMATION SECTION ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">ℹ️ About This Tool</div>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("""
    **Model Details:**
    - **Type:** OLS Regression with HAC Standard Errors
    - **Estimation:** Newey-West (3 lags)
    - **R² (In-sample):** {:.4f}
    - **Adjusted R²:** {:.4f}
    - **Observations:** {} months (Oct 2018 – Jun 2025)
    """.format(model.rsquared, model.rsquared_adj, len(df)))

with col_info2:
    st.markdown("""
    **Variables Included:**
    - WPI growth (cost-push channel)
    - IIP growth (demand-pull channel)
    - Import growth (supply augmentation)
    - Government borrowing (monetary/fiscal)
    - Lagged food inflation (persistence)
    - Lagged inflation (AR effects)
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;margin-top:1rem'>"
    "<b>India Inflation Prediction Tool</b> | OLS Regression with HAC Standard Errors | "
    "Data: RBI DBIE, MOSPI, OEA | Oct 2018 – Jun 2025<br>"
    "<i>Macroeconomic Determinants of India's CPI Inflation</i>"
    "</div>",
    unsafe_allow_html=True
)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inflation Prediction Tool",
    page_icon="📈",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1F4E79;
        border-bottom: 3px solid #2E74B5; padding-bottom: 0.5rem; margin-bottom: 0.2rem;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1F4E79;
        border-left: 4px solid #2E74B5; padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("regression_final.csv")
    df['Period'] = pd.to_datetime(df['Period'])
    df = df.sort_values('Period').reset_index(drop=True)
    if 'inflation_lag1' not in df.columns:
        df['inflation_lag1'] = df['Inflation_Rate_YoY'].shift(1)
        df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

IVs = ['WPI_yoy','IIP_yoy','Imports_yoy','MarketBorrowing_yoy',
       'CPI_food_lag1_yoy','inflation_lag1']
DV  = 'Inflation_Rate_YoY'

# ── FIT MODEL ─────────────────────────────────────────────────────────────────
@st.cache_data
def fit_model(data):
    X = sm.add_constant(data[IVs])
    y = data[DV]
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return model

model = fit_model(df)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION TOOL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">Inflation Prediction Tool</div>', unsafe_allow_html=True)
st.markdown("Enter macroeconomic values to predict India's CPI inflation rate.")

col1, col2, col3 = st.columns(3)
with col1:
    wpi        = st.number_input("WPI YoY % (Wholesale Price Growth)",     min_value=-20.0, max_value=30.0,  value=3.5,  step=0.1)
    iip        = st.number_input("IIP YoY % (Industrial Production Growth)",min_value=-30.0, max_value=50.0,  value=5.0,  step=0.1)
with col2:
    imports    = st.number_input("Imports YoY % (Import Value Growth)",    min_value=-60.0, max_value=120.0, value=8.0,  step=0.5)
    mkt_borrow = st.number_input("Market Borrowing YoY %",                 min_value=-100.0,max_value=150.0, value=10.0, step=1.0)
with col3:
    food_lag   = st.number_input("CPI Food Lag1 YoY % (Prev month food inflation)", min_value=-5.0, max_value=20.0, value=5.5, step=0.1)
    inf_lag    = st.number_input("Inflation Lag1 % (Prev month CPI inflation)",     min_value=-2.0, max_value=12.0, value=4.5, step=0.1)

if st.button("🔮 Predict Inflation", type="primary"):
    input_vals = np.array([[1.0, wpi, iip, imports, mkt_borrow, food_lag, inf_lag]])
    prediction = model.params @ input_vals[0]
    ci_lo_pred = prediction - 1.96 * np.sqrt(model.scale)
    ci_hi_pred = prediction + 1.96 * np.sqrt(model.scale)

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Predicted CPI Inflation", f"{prediction:.2f}%")
    col_r2.metric("95% CI Lower",            f"{ci_lo_pred:.2f}%")
    col_r3.metric("95% CI Upper",            f"{ci_hi_pred:.2f}%")

    if prediction <= 4.0:
        st.success(f"✅ Predicted inflation {prediction:.2f}% is within RBI target (≤ 4%)")
    elif prediction <= 6.0:
        st.warning(f"⚠️ Predicted inflation {prediction:.2f}% is above RBI target but within tolerance band (4–6%)")
    else:
        st.error(f"❌ Predicted inflation {prediction:.2f}% exceeds RBI upper tolerance (6%)")

    st.markdown('<div class="section-header">Contribution of Each Variable</div>', unsafe_allow_html=True)
    contrib_vars = IVs
    inputs_no_const = [wpi, iip, imports, mkt_borrow, food_lag, inf_lag]
    contribs = [model.params[v] * inp for v, inp in zip(contrib_vars, inputs_no_const)]
    contrib_df = pd.DataFrame({
        'Variable':     contrib_vars,
        'Input Value':  [round(x, 2) for x in inputs_no_const],
        'Coefficient':  [round(model.params[v], 4) for v in contrib_vars],
        'Contribution': [round(c, 4) for c in contribs]
    })
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ['#1D9E75' if c > 0 else '#E24B4A' for c in contribs]
    ax.barh(contrib_vars, contribs, color=bar_colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Contribution to Predicted Inflation (%)')
    ax.set_title('Variable Contributions to Prediction', fontweight='bold')
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem'>"
    "India Inflation Prediction Tool | OLS Regression with HAC Standard Errors | "
    "Data: RBI DBIE, MOSPI, OEA | Oct 2018 – Jun 2025"
    "</div>",
    unsafe_allow_html=True
)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{model.rsquared_adj:.3f}</div><div class="metric-lab">Adjusted R²</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-lab">Monthly Observations</div></div>', unsafe_allow_html=True)
    with col4:
        sig_count = sum(1 for v in IVs if model.pvalues[v] < 0.05)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{sig_count} / {len(IVs)}</div><div class="metric-lab">Significant Variables</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Inflation Rate Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Period'], df[DV], color='#1F4E79', linewidth=2, label='Actual CPI Inflation')
    ax.plot(df['Period'], model.fittedvalues, color='#E05A2B', linewidth=1.5,
            linestyle='--', label='Model Fitted Values')
    ax.fill_between(df['Period'], df[DV], model.fittedvalues,
                    alpha=0.1, color='#E05A2B')
    ax.axhline(4, color='green', linestyle=':', linewidth=1.2, label='RBI Target (4%)')
    ax.axhline(6, color='red',   linestyle=':', linewidth=1.0, label='Upper Tolerance (6%)')
    ax.set_ylabel('Inflation Rate (YoY %)')
    ax.set_title('India CPI Inflation — Actual vs Model Fitted', fontweight='bold')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Problem Statement</div>', unsafe_allow_html=True)
    st.info("""
    **Research Question:** Do macroeconomic variables including wholesale price movements, industrial production,
    import dynamics, government market borrowing, lagged food inflation, and prior inflation levels collectively
    and individually explain the variation in India's monthly CPI inflation between October 2018 and June 2025?
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Significant Findings</div>', unsafe_allow_html=True)
        findings = {
            'WPI_yoy':         ('Cost-push channel confirmed', '0.1369', '***'),
            'IIP_yoy':         ('Demand-pull channel confirmed', '0.0222', '**'),
            'Imports_yoy':     ('Supply augmentation channel', '-0.0197', '***'),
            'inflation_lag1':  ('Strong inflation persistence', '0.5618', '***'),
        }
        for var, (desc, coef, sig) in findings.items():
            st.markdown(f'<div class="insight-box"><b>{var}</b> ({coef}) — {desc} <span class="sig-high">{sig}</span></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="section-header">Model Specification</div>', unsafe_allow_html=True)
        st.markdown("""
        **Dependent Variable (DV):**
        - `Inflation_Rate_YoY` — CPI Year-on-Year %

        **Independent Variables (IVs):**
        - `WPI_yoy` — Wholesale Price Index growth
        - `IIP_yoy` — Industrial Production growth
        - `Imports_yoy` — Import value growth
        - `MarketBorrowing_yoy` — Govt. borrowing growth
        - `CPI_food_lag1_yoy` — Lagged food inflation
        - `inflation_lag1` — Lagged inflation (AR term)

        **Method:** OLS with HAC (Newey-West, 3 lags)
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[IVs + [DV]].describe().round(3), use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    corr = df[IVs + [DV]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 9})
    ax.set_title('Correlation Matrix — All Variables', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Variable Distributions (Boxplots)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    all_vars = IVs + [DV]
    for i, var in enumerate(all_vars):
        axes[i].boxplot(df[var].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='#B5D4F4', color='#1F4E79'),
                        medianprops=dict(color='#E05A2B', linewidth=2),
                        whiskerprops=dict(color='#1F4E79'),
                        capprops=dict(color='#1F4E79'),
                        flierprops=dict(marker='o', color='#888', markersize=4))
        axes[i].set_title(var, fontsize=9, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
        sns.despine(ax=axes[i])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Time Series — All Variables</div>', unsafe_allow_html=True)
    selected = st.multiselect("Select variables to plot:", IVs + [DV], default=[DV, 'WPI_yoy'])
    if selected:
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#1F4E79','#E05A2B','#0F6E56','#BA7517','#534AB7','#993C1D','#185FA5']
        for i, var in enumerate(selected):
            ax.plot(df['Period'], df[var], label=var, color=colors[i % len(colors)], linewidth=1.8)
        ax.set_ylabel('YoY %')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIF CHECK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 VIF Check":
    st.markdown('<div class="main-title">Multicollinearity Check — VIF</div>', unsafe_allow_html=True)
    st.info("Variance Inflation Factor (VIF) measures multicollinearity. Threshold: VIF < 10 acceptable, > 10 concern.")

    X_vif = df[IVs].copy()
    vif_df = pd.DataFrame({
        'Variable': IVs,
        'VIF': [round(variance_inflation_factor(X_vif.values, i), 3) for i in range(len(IVs))]
    })
    vif_df['Status'] = vif_df['VIF'].apply(
        lambda v: '✅ Acceptable' if v < 10 else ('⚠️ High — noted' if v < 25 else '❌ Very High'))

    st.dataframe(vif_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#E24B4A' if v > 10 else '#1D9E75' for v in vif_df['VIF']]
    bars = ax.barh(vif_df['Variable'], vif_df['VIF'], color=colors)
    ax.axvline(10, color='red', linestyle='--', linewidth=1.5, label='Threshold (VIF=10)')
    for bar, val in zip(bars, vif_df['VIF']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)
    ax.set_xlabel('VIF Value')
    ax.set_title('VIF — Multicollinearity Check', fontweight='bold')
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
    - **WPI_yoy, IIP_yoy, Imports_yoy, MarketBorrowing_yoy** — All VIF < 10 ✅
    - **CPI_food_lag1_yoy** — VIF = 12.6 ⚠️ Reflects structural link between food and headline CPI
    - **inflation_lag1** — VIF = 20.1 ⚠️ Expected for autoregressive terms; theoretically motivated
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REGRESSION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Regression Results":
    st.markdown('<div class="main-title">OLS Regression Results</div>', unsafe_allow_html=True)
    st.markdown("**Model:** Dynamic OLS with HAC Standard Errors (Newey-West, 3 lags)")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("R²",           f"{model.rsquared:.4f}")
    col2.metric("Adj. R²",      f"{model.rsquared_adj:.4f}")
    col3.metric("F-statistic",  f"{model.fvalue:.2f}")
    col4.metric("AIC",          f"{model.aic:.1f}")
    col5.metric("Observations", f"{int(model.nobs)}")

    st.markdown('<div class="section-header">Coefficient Table</div>', unsafe_allow_html=True)
    coef_data = []
    for var in ['const'] + IVs:
        p = model.pvalues[var]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '—'))
        coef_data.append({
            'Variable':    var,
            'Coefficient': round(model.params[var], 4),
            'Std Error':   round(model.bse[var], 4),
            'z-stat':      round(model.tvalues[var], 3),
            'p-value':     round(p, 4),
            'CI Lower':    round(model.conf_int().loc[var, 0], 4),
            'CI Upper':    round(model.conf_int().loc[var, 1], 4),
            'Sig':         sig
        })
    coef_df = pd.DataFrame(coef_data)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption("*** p<0.001  ** p<0.01  * p<0.05  — not significant")

    st.markdown('<div class="section-header">Coefficient Plot</div>', unsafe_allow_html=True)
    plot_vars = IVs
    coefs  = [model.params[v] for v in plot_vars]
    ci_lo  = [model.conf_int().loc[v, 0] for v in plot_vars]
    ci_hi  = [model.conf_int().loc[v, 1] for v in plot_vars]
    errors = [[c - l for c, l in zip(coefs, ci_lo)],
              [h - c for c, h in zip(coefs, ci_hi)]]
    colors = ['#1D9E75' if model.pvalues[v] < 0.05 else '#B4B2A9' for v in plot_vars]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(plot_vars))
    ax.barh(y_pos, coefs, color=colors, alpha=0.85, height=0.55)
    ax.errorbar(coefs, y_pos, xerr=errors, fmt='none',
                color='#333', capsize=4, linewidth=1.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_vars, fontsize=10)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('OLS Coefficients with 95% Confidence Intervals\n(Green = significant, Grey = not significant)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Hypothesis Test Results</div>', unsafe_allow_html=True)
    hyp_data = [
        ('H₁', 'WPI_yoy → + CPI',          model.pvalues['WPI_yoy'],           'Rejected ✅'),
        ('H₂', 'IIP_yoy → + CPI',           model.pvalues['IIP_yoy'],           'Rejected ✅'),
        ('H₃', 'Imports_yoy → − CPI',       model.pvalues['Imports_yoy'],       'Rejected ✅'),
        ('H₄', 'MarketBorrowing → + CPI',   model.pvalues['MarketBorrowing_yoy'],'Not Rejected ❌'),
        ('H₅', 'CPI_food_lag1 → + CPI',     model.pvalues['CPI_food_lag1_yoy'], 'Not Rejected ❌'),
        ('H₆', 'inflation_lag1 → + CPI',    model.pvalues['inflation_lag1'],    'Rejected ✅'),
    ]
    hyp_df = pd.DataFrame(hyp_data, columns=['Hyp', 'Statement', 'p-value', 'H₀ Decision'])
    hyp_df['p-value'] = hyp_df['p-value'].round(4)
    st.dataframe(hyp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Diagnostics":
    st.markdown('<div class="main-title">Regression Diagnostics</div>', unsafe_allow_html=True)

    residuals    = model.resid
    fitted_vals  = model.fittedvalues
    dw_stat      = durbin_watson(residuals)
    X_full       = sm.add_constant(df[IVs])
    bp           = het_breuschpagan(residuals, X_full)
    jb           = stats.jarque_bera(residuals)
    sw           = stats.shapiro(residuals)
    lb           = acorr_ljungbox(residuals, lags=[6, 12], return_df=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Durbin-Watson",      f"{dw_stat:.3f}",  delta="Mild autocorr")
    col2.metric("Breusch-Pagan p",    f"{bp[1]:.4f}",    delta="No heteroscedasticity ✅")
    col3.metric("Jarque-Bera p",      f"{float(jb[1]):.6f}", delta="Non-normal (food spikes)")
    col4.metric("Ljung-Box lag6 p",   f"{float(lb['lb_pvalue'].iloc[0]):.4f}", delta="HAC applied")

    st.markdown('<div class="section-header">Diagnostic Plots</div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Actual vs Fitted
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['Period'], df[DV],       color='#1F4E79', linewidth=2, label='Actual')
    ax1.plot(df['Period'], fitted_vals,  color='#E05A2B', linewidth=1.5, linestyle='--', label='Fitted')
    ax1.set_title('Actual vs Fitted Inflation', fontweight='bold')
    ax1.set_ylabel('Inflation Rate (YoY %)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax1)

    # Residuals vs Fitted
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(fitted_vals, residuals, alpha=0.6, color='#378ADD', s=30)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted', fontweight='bold')
    ax2.grid(alpha=0.3)
    sns.despine(ax=ax2)

    # Q-Q Plot
    ax3 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax3.scatter(osm, osr, alpha=0.6, color='#378ADD', s=30)
    x_line = np.array([min(osm), max(osm)])
    ax3.plot(x_line, slope * x_line + intercept, color='red', linewidth=1.5)
    ax3.set_xlabel('Theoretical Quantiles')
    ax3.set_ylabel('Sample Quantiles')
    ax3.set_title('Q-Q Plot of Residuals', fontweight='bold')
    ax3.grid(alpha=0.3)
    sns.despine(ax=ax3)

    st.pyplot(fig)
    plt.close()

    # Residual histogram separately
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    axes2[0].hist(residuals, bins=15, color='#B5D4F4', edgecolor='white')
    axes2[0].set_title('Residual Distribution', fontweight='bold')
    axes2[0].set_xlabel('Residual')
    axes2[0].set_ylabel('Frequency')
    sns.despine(ax=axes2[0])

    axes2[1].plot(df['Period'], residuals, color='#888', linewidth=1)
    axes2[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes2[1].set_title('Residuals Over Time', fontweight='bold')
    axes2[1].set_ylabel('Residual')
    axes2[1].tick_params(axis='x', rotation=30)
    axes2[1].grid(axis='y', alpha=0.3)
    sns.despine(ax=axes2[1])
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('<div class="section-header">Diagnostic Summary</div>', unsafe_allow_html=True)
    diag_df = pd.DataFrame([
        ['Durbin-Watson',          f'{dw_stat:.3f}',                  '1.32 — mild autocorrelation',   'HAC errors applied ✅'],
        ['Breusch-Pagan',          f'p = {bp[1]:.4f}',                'p > 0.05 — no heteroscedasticity','Assumption satisfied ✅'],
        ['Jarque-Bera',            f'p = {float(jb[1]):.6f}',         'p < 0.05 — non-normal residuals','Food spike outliers (stated limitation)'],
        ['Shapiro-Wilk',           f'p = {float(sw[1]):.4f}',         'Mild non-normality',             'HAC robust; CLT applies (n=77)'],
        ['Ljung-Box (lag 6)',      f'p = {float(lb["lb_pvalue"].iloc[0]):.4f}', 'Serial correlation',  'HAC correction applied ✅'],
        ['Ljung-Box (lag 12)',     f'p = {float(lb["lb_pvalue"].iloc[1]):.4f}', 'Serial correlation',  'HAC correction applied ✅'],
    ], columns=['Test', 'Statistic', 'Finding', 'Action'])
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.info("Comparing OLS vs Ridge vs Lasso vs Random Forest using 5-fold TimeSeriesSplit cross-validation.")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df[IVs])
    y_cv   = df[DV].values
    tscv   = TimeSeriesSplit(n_splits=5)

    with st.spinner("Running cross-validation..."):
        cv_results = {}
        for name, mdl in [
            ('OLS',           LinearRegression()),
            ('Ridge',         Ridge(alpha=1.0)),
            ('Lasso',         Lasso(alpha=0.1, max_iter=10000)),
            ('Random Forest', RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))
        ]:
            scores = cross_val_score(mdl, X_sc, y_cv, cv=tscv, scoring='r2')
            cv_results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores.tolist()}

    col1, col2, col3, col4 = st.columns(4)
    for col, (name, res) in zip([col1, col2, col3, col4], cv_results.items()):
        col.metric(name, f"{res['mean']:.4f}", delta=f"±{res['std']:.4f}")

    cv_df = pd.DataFrame([
        {'Model': k, 'Mean CV R²': round(v['mean'], 4), 'Std Dev': round(v['std'], 4),
         'In-sample R²': round(model.rsquared, 4) if k == 'OLS' else '—',
         'Best for': 'Explanation' if k == 'OLS' else ('Generalisation' if k == 'Ridge' else ('Shrinkage' if k == 'Lasso' else 'Non-linear'))}
        for k, v in cv_results.items()
    ])
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(cv_results.keys())
    means  = [cv_results[n]['mean'] for n in names]
    stds   = [cv_results[n]['std']  for n in names]
    colors = ['#1D9E75' if m == max(means) else '#B5D4F4' for m in means]
    bars   = ax.bar(names, means, color=colors, edgecolor='white', width=0.5)
    ax.errorbar(names, means, yerr=stds, fmt='none', color='#333', capsize=5, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.set_title('Cross-Validation R² — Model Comparison', fontweight='bold')
    ax.set_ylabel('Mean CV R²')
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Random Forest Feature Importance</div>', unsafe_allow_html=True)
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    rf.fit(X_sc, y_cv)
    imp_df = pd.DataFrame({'Variable': IVs, 'Importance': rf.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors2 = ['#1F4E79' if v > imp_df['Importance'].median() else '#B5D4F4' for v in imp_df['Importance']]
    ax2.barh(imp_df['Variable'], imp_df['Importance'], color=colors2)
    ax2.set_title('RF Feature Importance', fontweight='bold')
    ax2.set_xlabel('Importance Score')
    sns.despine(ax2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("""
    <div class="warn-box">
    <b>Note on negative CV R²:</b> OLS shows negative mean CV R² due to small sample size (77 obs).
    With 5-fold TimeSeriesSplit, each training fold has ~45–55 rows — insufficient for stable
    out-of-sample prediction with an autoregressive term. This is a known constraint of monthly
    macroeconomic datasets, not a model failure. OLS remains the best model for <i>explanation</i>.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PREDICTION TOOL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction Tool":
    st.markdown('<div class="main-title">Inflation Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown("Enter macroeconomic values to predict India's CPI inflation rate.")

    col1, col2, col3 = st.columns(3)
    with col1:
        wpi        = st.number_input("WPI YoY % (Wholesale Price Growth)",     min_value=-20.0, max_value=30.0,  value=3.5,  step=0.1)
        iip        = st.number_input("IIP YoY % (Industrial Production Growth)",min_value=-30.0, max_value=50.0,  value=5.0,  step=0.1)
    with col2:
        imports    = st.number_input("Imports YoY % (Import Value Growth)",    min_value=-60.0, max_value=120.0, value=8.0,  step=0.5)
        mkt_borrow = st.number_input("Market Borrowing YoY %",                 min_value=-100.0,max_value=150.0, value=10.0, step=1.0)
    with col3:
        food_lag   = st.number_input("CPI Food Lag1 YoY % (Prev month food inflation)", min_value=-5.0, max_value=20.0, value=5.5, step=0.1)
        inf_lag    = st.number_input("Inflation Lag1 % (Prev month CPI inflation)",     min_value=-2.0, max_value=12.0, value=4.5, step=0.1)

    if st.button("🔮 Predict Inflation", type="primary"):
        input_vals = np.array([[1.0, wpi, iip, imports, mkt_borrow, food_lag, inf_lag]])
        prediction = model.params @ input_vals[0]
        ci_lo_pred = prediction - 1.96 * np.sqrt(model.scale)
        ci_hi_pred = prediction + 1.96 * np.sqrt(model.scale)

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Predicted CPI Inflation", f"{prediction:.2f}%")
        col_r2.metric("95% CI Lower",            f"{ci_lo_pred:.2f}%")
        col_r3.metric("95% CI Upper",            f"{ci_hi_pred:.2f}%")

        if prediction <= 4.0:
            st.success(f"✅ Predicted inflation {prediction:.2f}% is within RBI target (≤ 4%)")
        elif prediction <= 6.0:
            st.warning(f"⚠️ Predicted inflation {prediction:.2f}% is above RBI target but within tolerance band (4–6%)")
        else:
            st.error(f"❌ Predicted inflation {prediction:.2f}% exceeds RBI upper tolerance (6%)")

        st.markdown('<div class="section-header">Contribution of Each Variable</div>', unsafe_allow_html=True)
        contrib_vars = IVs
        inputs_no_const = [wpi, iip, imports, mkt_borrow, food_lag, inf_lag]
        contribs = [model.params[v] * inp for v, inp in zip(contrib_vars, inputs_no_const)]
        contrib_df = pd.DataFrame({
            'Variable':     contrib_vars,
            'Input Value':  [round(x, 2) for x in inputs_no_const],
            'Coefficient':  [round(model.params[v], 4) for v in contrib_vars],
            'Contribution': [round(c, 4) for c in contribs]
        })
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = ['#1D9E75' if c > 0 else '#E24B4A' for c in contribs]
        ax.barh(contrib_vars, contribs, color=bar_colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Contribution to Predicted Inflation (%)')
        ax.set_title('Variable Contributions to Prediction', fontweight='bold')
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem'>"
    "India Inflation Analytics Engine | OLS Regression with HAC Standard Errors | "
    "Data: RBI DBIE, MOSPI, OEA | Oct 2018 – Jun 2025"
    "</div>",
    unsafe_allow_html=True
)
