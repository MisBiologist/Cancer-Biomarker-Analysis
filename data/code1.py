import pandas as pd

# 1. Load the dataset
data = pd.read_csv("data.csv")

# 2. Explore the data
print("First 5 rows of the dataset:")
print(data.csv.head())  # Shows first 5 rows

print("\nList of columns:")
print(data.csv.columns)  # Shows all column names

print("\nCount of cancer types (Malignant/Benign):")
print(data.csv['diagnosis'].value_counts())  # Shows how many M (malignant) and B (benign) cases