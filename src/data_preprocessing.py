import pandas as pd

# Load the data
df = pd.read_csv('data/breast_cancer_data.csv')

# Simple cleaning: Drop missing values
df.dropna(inplace=True)

# Save the cleaned data
df.to_csv('data/cleaned_data.csv', index=False)
print("✅ Cleaned data saved to data/cleaned_data.csv")