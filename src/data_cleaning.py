import pandas as pd

# Load raw dataset
file_path = r"C:\Users\User\OneDrive\Mobile data\OneDrive\Attachments\GrassInternship_Project\data\raw\archive\SuperStoreOrders.csv"
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# Convert date columns
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)
df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce", dayfirst=True)

# Convert sales from object to numeric
df["sales"] = (
    df["sales"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

# Remove duplicate rows if any
df = df.drop_duplicates()

# Check data after cleaning
print("\nData types after cleaning:")
print(df.dtypes)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned shape:", df.shape)

# Save cleaned dataset
output_path = "data/processed/clean_superstore.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")