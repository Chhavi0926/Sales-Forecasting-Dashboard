import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/processed/clean_superstore.csv")

# Convert order_date into date format
df["order_date"] = pd.to_datetime(df["order_date"])

# 1. Monthly Sales
# 1. Monthly Sales
monthly_sales = df.groupby(df["order_date"].dt.to_period("M"))["sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Sales Trend", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)

plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()


# 2. Sales by Category
category_sales = df.groupby("category")["sales"].sum()

print("\nSales by Category:")
print(category_sales)

category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()


# 3. Sales by Region
region_sales = df.groupby("region")["sales"].sum()

print("\nSales by Region:")
print(region_sales)

region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()


# 4. Profit by Category
category_profit = df.groupby("category")["profit"].sum()

print("\nProfit by Category:")
print(category_profit)

category_profit.plot(kind="bar")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.show()


print("\nEDA completed successfully!")