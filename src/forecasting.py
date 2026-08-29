import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------------------
# 1. Load Data
# -----------------------------------------

df = pd.read_csv("data/processed/clean_superstore.csv")

df["order_date"] = pd.to_datetime(df["order_date"])


# -----------------------------------------
# 2. Create Monthly Sales
# -----------------------------------------

monthly_sales = (
    df.groupby(df["order_date"].dt.to_period("M"))["sales"]
    .sum()
    .reset_index()
)

monthly_sales["order_date"] = (
    monthly_sales["order_date"].dt.to_timestamp()
)


print("\nMonthly Sales Data:")
print(monthly_sales)


# -----------------------------------------
# 3. Create Forecasting Features
# -----------------------------------------

monthly_sales["month_number"] = np.arange(len(monthly_sales))

monthly_sales["month"] = monthly_sales["order_date"].dt.month

# Previous year's sales
monthly_sales["lag_12"] = monthly_sales["sales"].shift(12)

# Seasonal features
monthly_sales["month_sin"] = np.sin(
    2 * np.pi * monthly_sales["month"] / 12
)

monthly_sales["month_cos"] = np.cos(
    2 * np.pi * monthly_sales["month"] / 12
)


# Remove rows where previous year's sales is not available
model_data = monthly_sales.dropna().copy()


# -----------------------------------------
# 4. Train-Test Split
# -----------------------------------------

train_data = model_data.iloc[:-6]
test_data = model_data.iloc[-6:]


features = [
    "month_number",
    "month_sin",
    "month_cos",
    "lag_12"
]


X_train = train_data[features]
y_train = train_data["sales"]

X_test = test_data[features]
y_test = test_data["sales"]


# -----------------------------------------
# 5. Train Improved Model
# -----------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# -----------------------------------------
# 6. Predict Test Data
# -----------------------------------------

test_predictions = model.predict(X_test)


# -----------------------------------------
# 7. Model Evaluation
# -----------------------------------------

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


print("\nImproved Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# -----------------------------------------
# 8. Test Prediction Graph
# -----------------------------------------

plt.figure(figsize=(12, 6))

# Actual test sales
plt.plot(
    test_data["order_date"],
    y_test,
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

# Predicted test sales
plt.plot(
    test_data["order_date"],
    test_predictions,
    marker="o",
    linestyle="--",
    linewidth=2,
    label="Predicted Sales"
)

plt.title(
    "Test Data - Actual vs Predicted Sales",
    fontsize=16
)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Sales", fontsize=12)

plt.legend()

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# -----------------------------------------
# 9. Future 6-Month Forecast
# -----------------------------------------

last_date = monthly_sales["order_date"].iloc[-1]

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)


future_data = pd.DataFrame({
    "order_date": future_dates
})


future_data["month_number"] = np.arange(
    len(monthly_sales),
    len(monthly_sales) + 6
)

future_data["month"] = future_data["order_date"].dt.month


# Previous year's sales for future months
future_data["lag_12"] = [
    monthly_sales.loc[
        monthly_sales["order_date"] ==
        date - pd.DateOffset(years=1),
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


# -----------------------------------------
# 10. Predict Future Sales
# -----------------------------------------

future_predictions = model.predict(
    future_data[features]
)


forecast_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": future_predictions
})


print("\nFuture Sales Forecast:")
print(forecast_df)


# -----------------------------------------
# 11. Actual vs Predicted Graph
# -----------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["order_date"],
    monthly_sales["sales"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

plt.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    linewidth=2,
    label="Forecasted Sales"
)

plt.title(
    "Sales Forecast - Actual vs Predicted",
    fontsize=16
)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Sales", fontsize=12)

plt.legend()

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()