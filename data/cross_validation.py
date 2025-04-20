import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('data.csv')
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Initialize model
model = RandomForestClassifier()

# Cross-validate
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-Validation Accuracy: {scores.mean()*100:.2f}% (±{scores.std()*100:.2f}%)")