# code3.py
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # <--- ADD THIS LINE
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('data.csv')
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'cancer_model.pkl')  # Save the model
print("Model saved as cancer_model.pkl!")

# Save feature importance plot
plt.barh(X.columns[:10], model.feature_importances_[:10])
plt.savefig('biomarkers.png')
plt.close()
import plotly.express as px

# Example: Interactive Heatmap
fig = px.imshow(
    data.corr(),  # Replace `data` with your DataFrame
    labels=dict(x="Features", y="Features", color="Correlation"),
    color_continuous_scale='RdBu',  # Red-Blue color scale
    title="Interactive Correlation Heatmap (Zoom In!)",
    width=1000,
    height=1000
)

# Save as HTML (click and zoom!)
fig.write_html("heatmap_interactive.html")

# Optional: Display in Jupyter Notebook
# fig.show()
