import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Dataset: VAR1, VAR2, and their classifications
data = np.array([
    [0.1, 0.6], 
    [0.15, 0.71], 
    [0.08, 0.9], 
    [0.16, 0.85], 
    [0.2, 0.3], 
    [0.25, 0.5], 
    [0.24, 0.1], 
    [0.3, 0.2], 
    [0.85, 0.8]
])

# Standardize the data (K-means works better with scaled data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# Predict the cluster for the new data point (VAR1=0.906, VAR2=0.606)
new_point = scaler.transform([[0.906, 0.606]])  # Scale the new point
predicted_cluster = kmeans.predict(new_point)

# Inverse transform the centroids to the original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Plotting the clusters and the new point
plt.figure(figsize=(8, 6))

# Plot the data points, colored by their cluster
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', s=100, label='Data points')

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Plot the new point
plt.scatter(new_point[:, 0], new_point[:, 1], c='blue', marker='*', s=200, label='New point (0.906, 0.606)')

# Add labels and legend
plt.xlabel('VAR1')
plt.ylabel('VAR2')
plt.title('K-means Clustering with 3 Clusters')
plt.legend()
plt.grid(True)
plt.show()

# Print the results
print("Centroids of the clusters:")
print(centroids)
print(f"Predicted cluster for VAR1=0.906, VAR2=0.606: {predicted_cluster[0]}")
