import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
file_path = 'screen_coordinates.xlsx'
data = pd.read_excel(file_path)

# Drop rows with NaN values in 'PlaneSize'
data.dropna(subset=['PlaneSize'], inplace=True)

# Ensure 'Group' column is correctly set as category or numeric type
data['Group'] = data['Group'].astype(str)

# Add 'NumberOfNiches' column by counting the occurrences per 'Group' and 'Category'
data['NumberOfNiches'] = data.groupby(['Group', 'Category'])['X'].transform('count')

# Aggregate data to have one row per ovary
unique_data = data.drop_duplicates(subset=['Group', 'Category', 'PlaneSize']).copy()

# Define bins and labels based on desired cutoffs
bins = [0, 1600, 2400, np.inf]
labels = ['Small', 'Medium', 'Large']

# Apply the bins and labels to the data using .loc to avoid warnings
unique_data.loc[:, 'PlaneSizeRange'] = pd.cut(unique_data['PlaneSize'], bins=bins, labels=labels, right=False)

# Check the distribution of data within each bin
bin_counts = unique_data['PlaneSizeRange'].value_counts().sort_index()
print(f"\nCounts per bin:\n{bin_counts}")

# Clustering validation: Use K-Means clustering to determine natural groupings with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
unique_data.loc[:, 'Cluster'] = kmeans.fit_predict(unique_data[['PlaneSize']])
cluster_centers = kmeans.cluster_centers_.flatten()

# Create a mapping for cluster labels
cluster_labels = {}
sorted_centers = np.sort(cluster_centers)
labels_kmeans = ['Small', 'Medium', 'Large']

for i, center in enumerate(sorted_centers):
    cluster_labels[np.argmin(np.abs(cluster_centers - center))] = labels_kmeans[i]

unique_data.loc[:, 'Cluster'] = unique_data['Cluster'].map(cluster_labels)

# Visualize the distribution of PlaneSize with class boundaries and cluster centers
plt.figure(figsize=(10, 6))
plt.hist(unique_data['PlaneSize'], bins=30, edgecolor='black', alpha=0.6, label='PlaneSize Distribution')
for boundary in bins[1:-1]:
    plt.axvline(boundary, color='r', linestyle='--', label=f'Bin Boundary ({boundary:.2f})')
for center in sorted_centers:
    plt.axvline(center, color='g', linestyle='--', label=f'Cluster Center ({center:.2f})')
plt.xlabel('PlaneSize')
plt.ylabel('Frequency')
plt.title('Comparison of PlaneSize Distribution with Bins and Cluster Centers')
plt.legend()
plt.show()

# Print cluster centers
print(f"\nCluster centers:\n{sorted_centers}")

# Scatter plot to visualize PlaneSize and NumberOfNiches
plt.figure(figsize=(10, 6))
for cluster in labels:
    subset = unique_data[unique_data['PlaneSizeRange'] == cluster]
    plt.scatter(subset['NumberOfNiches'], subset['PlaneSize'], label=cluster)
plt.xlabel('Number of Niches')
plt.ylabel('PlaneSize')
plt.title('PlaneSize vs. Number of Niches by Size Range')
plt.legend()
plt.show()
