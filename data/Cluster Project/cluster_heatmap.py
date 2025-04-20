import plotly.express as px
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.figure_factory as ff

# Example data (replace with your dataset)
data = np.random.rand(10, 10)  # Random 10x10 data matrix
columns = ['Feature ' + str(i) for i in range(1, 11)]
index = ['Sample ' + str(i) for i in range(1, 11)]

# Create a DataFrame (replace this with your actual dataset)
df = pd.DataFrame(data, columns=columns, index=index)

# Convert the pandas index and columns to lists
x_vals = df.columns.tolist()  # Convert to list
y_vals = df.index.tolist()    # Convert to list

# Perform hierarchical clustering
row_linkage = linkage(df, method='average', metric='euclidean')
col_linkage = linkage(df.T, method='average', metric='euclidean')

# Get dendrogram coordinates (but not directly using dendrogram in Plotly)
fig = ff.create_annotated_heatmap(
    z=df.values,
    x=x_vals,  # Use list of columns
    y=y_vals,  # Use list of rows
    colorscale='Viridis',
    colorbar=dict(title='Intensity'),
)

# Update the layout for better clarity
fig.update_layout(
    title='Clustered Heatmap for Breast Cancer Data',
    xaxis_title='Features',
    yaxis_title='Samples',
    width=1000,  # Adjust width for better clarity
    height=800,  # Adjust height for better clarity
    xaxis=dict(tickangle=45),  # Rotate x-axis labels
    yaxis=dict(tickangle=0),  # Rotate y-axis labels if needed
    margin=dict(l=100, r=100, t=100, b=100)  # Add margins to avoid crowding
)

# Save and open manually
fig.write_html("heatmap_result.html")
fig.show()