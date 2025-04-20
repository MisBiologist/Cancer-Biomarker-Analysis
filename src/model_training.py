# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data (make sure to use your preprocessed data)
import pandas as pd
data = pd.read_csv('processed_data.csv')

# Prepare the data
X = data.drop('target', axis=1)  # Replace 'target' with your label column
y = data['target']  # Replace 'target' with your label column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')