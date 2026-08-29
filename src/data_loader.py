import pandas as pd

# Full path to the dataset file

# file_path = "data/raw/archive/SuperStoreOrders.csv"
file_path = r"C:\Users\User\OneDrive\Mobile data\OneDrive\Attachments\GrassInternship_Project\data\raw\archive\SuperStoreOrders.csv"

# Load dataset
df = pd.read_csv(file_path)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Basic information about dataset
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())