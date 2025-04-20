import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load model and data
model = joblib.load('cancer_model.pkl')
data = pd.read_csv('data.csv')

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Predict
y_pred = model.predict(X)

# Print evaluation
print("\n🧪 Classification Report:")
print(classification_report(y, y_pred, target_names=['Benign (B)', 'Malignant (M)']))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y, y_pred))