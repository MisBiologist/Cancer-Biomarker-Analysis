import joblib
import pandas as pd

# Load your saved model
model = joblib.load('cancer_model.pkl')
print("✅ Model loaded successfully!")

# Load your data again (use same columns as training data)
data = pd.read_csv('data.csv')

# Drop the target column ('diagnosis') if it's still in there
X = data.drop('diagnosis', axis=1)

# Predict again
predictions = model.predict(X)

# Print first few predictions
print("🔍 Sample Predictions:", predictions[:10])