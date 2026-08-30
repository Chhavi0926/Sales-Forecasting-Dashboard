from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #2563EB, #0EA5E9, #38BDF8); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #B8C1CC;
    font-size: 17px;
    margin-bottom: 20px;
}

.badge {
    text-align: center;
    color: #2563EB;
    font-size: 14px;
    margin-bottom: 25px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCE7F7;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #2563EB;
    font-weight: 800;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #2563EB;
    padding: 12px 18px;
    margin: 25px 0 12px 0;
    border-left: 5px solid #38BDF8;
    background: #EFF6FF;
    border-radius: 10px;
}
.forecast-heading {
    font-size: 22px;
    font-weight: 700;
    color: #2563EB;
    background: linear-gradient(90deg, #EFF6FF, #F8FAFC);
    padding: 14px 18px;
    margin: 25px 0 12px 0;
    border-left: 5px solid #2563EB;
    border-radius: 10px;
}

.forecast-heading span {
    color: #64748B;
    font-size: 15px;
    font-weight: 500;
}

.forecast-heading {
    font-size: 22px;
    font-weight: 700;
    color: #2563EB;
    background: linear-gradient(90deg, #EFF6FF, #F8FAFC);
    padding: 14px 18px;
    margin: 25px 0 12px 0;
    border-left: 5px solid #2563EB;
    border-radius: 10px;
}

.forecast-heading span {
    color: #64748B;
    font-size: 15px;
    font-weight: 500;
}

[data-testid="stDataFrame"] {
    border: 1px solid #D9E7F7;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.08);
    margin-top: 10px;
}

[data-testid="stDataFrame"] div[role="row"]:hover {
    background-color: #EFF6FF !important;
}

</style>



<div class="main-title">
📊 Sales Forecasting Dashboard
</div>

<div class="subtitle">
Turn historical sales data into meaningful insights & future predictions
</div>

<div class="badge">
✨ AI • Data Analytics • Machine Learning
</div>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.title("📊 Sales Forecasting Dashboard")
st.write("Explore sales performance and future sales predictions.")


# =========================================
# LOAD DATA
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "clean_superstore.csv"
)

df["order_date"] = pd.to_datetime(
    df["order_date"]
)


# =========================================
# SIDEBAR FILTERS
# =========================================
st.sidebar.markdown("""
<div style="text-align:center; padding:10px 0 20px 0;">
    <div style="font-size:35px;">📊</div>
    <h2 style="margin:0;">Dashboard Controls</h2>
    <p style="color:#64748B; font-size:14px;">
        Filter & explore your sales data
    </p>
</div>
""", unsafe_allow_html=True)


# Year filter
years = sorted(df["order_date"].dt.year.unique())

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All"] + years
)

# Category filter
categories = sorted(df["category"].unique())

selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + categories
)

# Region filter
regions = sorted(df["region"].unique())

selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + regions
)


# Apply filters
filtered_df = df.copy()

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["order_date"].dt.year == selected_year
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["region"] == selected_region
    ]


# =========================================
# BASIC KPIs
# =========================================

total_sales = filtered_df["sales"].sum()

average_sales = filtered_df["sales"].mean()

total_profit = filtered_df["profit"].sum()

total_orders = filtered_df["order_id"].nunique()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "📈 Average Sales",
    f"${average_sales:,.0f}"
)

col3.metric(
    "💵 Total Profit",
    f"${total_profit:,.0f}"
)

col4.metric(
    "🛒 Total Orders",
    f"{total_orders:,}"
)
st.write("Filtered Rows:", len(filtered_df))

st.divider()


# =========================================
# MONTHLY SALES
# =========================================

monthly_sales = (
    filtered_df.groupby(
        filtered_df["order_date"].dt.to_period("M")
    )["sales"]
    .sum()
)

monthly_sales.index = pd.to_datetime(monthly_sales.index.astype(str))


st.markdown("""
<div class="section-title">
📈 Monthly Sales Trend
</div>
""", unsafe_allow_html=True)

fig1, ax1 = plt.subplots(
    figsize=(12, 5)
)

ax1.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

ax1.set_xlabel("Month")
ax1.set_ylabel("Sales")
ax1.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig1)


# =========================================
# CATEGORY & REGION
# =========================================

col1, col2 = st.columns(2)


with col1:

    st.markdown("""
<div class="section-title">
🛍️ Sales by Category
</div>
""", unsafe_allow_html=True)

    category_sales = (
    filtered_df.groupby("category")["sales"]
    .sum()
    .sort_values(ascending=False)
)

    fig2, ax2 = plt.subplots(
        figsize=(7, 5)
    )

    ax2.bar(
        category_sales.index,
        category_sales.values
    )

    ax2.set_xlabel("Category")
    ax2.set_ylabel("Sales")
    ax2.set_title("Sales by Category")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig2)


with col2:

    st.markdown("""
<div class="section-title">
🌎 Sales by Region
</div>
""", unsafe_allow_html=True)

    region_sales = (
        filtered_df.groupby("region")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig3, ax3 = plt.subplots(
        figsize=(7, 5)
    )

    ax3.bar(
        region_sales.index,
        region_sales.values
    )

    ax3.set_xlabel("Region")
    ax3.set_ylabel("Sales")
    ax3.set_title("Sales by Region")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig3)


# =========================================
# PROFIT BY CATEGORY
# =========================================

st.markdown("""
<div class="section-title">
💰 Profit Analysis
</div>
""", unsafe_allow_html=True)

category_profit = (
    filtered_df.groupby("category")["profit"]
    .sum()
    .sort_values(ascending=False)
)

fig4, ax4 = plt.subplots(
    figsize=(10, 5)
)

ax4.bar(
    category_profit.index,
    category_profit.values
)

ax4.set_xlabel("Category")
ax4.set_ylabel("Profit")

ax4.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

st.pyplot(fig4)


# =========================================
# FORECASTING
# =========================================

st.divider()

st.markdown("""
<div class="section-title">
📊 Sales Forecast
</div>
""", unsafe_allow_html=True)


forecast_data = monthly_sales.reset_index()

forecast_data.columns = [
    "order_date",
    "sales"
]

forecast_data["month_number"] = np.arange(
    len(forecast_data)
)

forecast_data["month"] = (
    forecast_data["order_date"].dt.month
)

forecast_data["lag_12"] = (
    forecast_data["sales"].shift(12)
)

forecast_data["month_sin"] = np.sin(
    2 * np.pi * forecast_data["month"] / 12
)

forecast_data["month_cos"] = np.cos(
    2 * np.pi * forecast_data["month"] / 12
)


model_data = forecast_data.dropna().copy()

if len(model_data) < 7:
    st.warning(
        "Not enough historical data is available for the selected filters. "
        "Please select broader filters or choose 'All' to generate the forecast."
    )
    st.stop()

features = [
    "month_number",
    "month_sin",
    "month_cos",
    "lag_12"
]


# =========================================
# TRAIN / TEST
# =========================================

train_data = model_data.iloc[:-6]

test_data = model_data.iloc[-6:]


X_train = train_data[features]

y_train = train_data["sales"]

X_test = test_data[features]

y_test = test_data["sales"]


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# =========================================
# MODEL EVALUATION
# =========================================

test_predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


col1, col2 = st.columns(2)

col1.metric(
    "🎯 MAE",
    f"{mae:,.0f}"
)

col2.metric(
    "📉 RMSE",
    f"{rmse:,.0f}"
)


# =========================================
# FUTURE FORECAST
# =========================================

last_date = forecast_data[
    "order_date"
].iloc[-1]


future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)


future_data = pd.DataFrame({
    "order_date": future_dates
})


future_data["month_number"] = np.arange(
    len(forecast_data),
    len(forecast_data) + 6
)

future_data["month"] = (
    future_data["order_date"].dt.month
)


future_data["lag_12"] = [
    forecast_data.loc[
        forecast_data["order_date"]
        == date - pd.DateOffset(years=1),
        "sales"
    ].iloc[0]
    for date in future_dates
]


future_data["month_sin"] = np.sin(
    2 * np.pi * future_data["month"] / 12
)

future_data["month_cos"] = np.cos(
    2 * np.pi * future_data["month"] / 12
)


future_predictions = model.predict(
    future_data[features]
)


# =========================================
# FORECAST GRAPH
# =========================================

fig5, ax5 = plt.subplots(
    figsize=(12, 6)
)

ax5.plot(
    forecast_data["order_date"],
    forecast_data["sales"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

ax5.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    linewidth=2,
    label="Forecasted Sales"
)

ax5.set_title(
    "Actual Sales vs Future Forecast"
)

ax5.set_xlabel("Month")

ax5.set_ylabel("Sales")

ax5.legend()

ax5.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig5)


# =========================================
# FORECAST TABLE
# =========================================


forecast_table = pd.DataFrame({
    "Month": future_dates.strftime("%B %Y"),
    "Predicted Sales": [
        f"${value:,.0f}"
        for value in future_predictions
    ]
})

st.markdown("""
<style>
.forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 18px;
}

.forecast-table th {
    background-color: #1565C0;
    color: white;
    font-weight: bold;
    text-align: center;
    padding: 14px;
    border: 1px solid black;
}

.forecast-table td {
    text-align: center;
    padding: 12px;
    background-color: white;
    color: black;
    border: 1px solid black;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    forecast_table.to_html(
        index=False,
        classes="forecast-table"
    ),
    unsafe_allow_html=True
)


st.success(
    "Dashboard loaded successfully! 🎉"
)