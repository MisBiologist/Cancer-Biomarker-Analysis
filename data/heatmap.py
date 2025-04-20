import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("data.csv")

# Drop target column if it exists
if 'diagnosis' in df.columns:
    df = df.drop('diagnosis', axis=1)

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Cancer Biomarkers')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()