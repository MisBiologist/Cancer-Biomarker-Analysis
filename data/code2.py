import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data.csv')
data = data.drop(['id', 'Unnamed: 32'], axis=1)

le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

print("\nDiagnosis counts after cleaning:")
print(y.value_counts())  # Output: B=357, M=212 (correct)