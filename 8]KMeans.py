import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

print("Centroids of the clusters:")
print(scaler.inverse_transform(kmeans.cluster_centers_))  # Inverse scale to get original centroids
print(f"Predicted cluster for VAR1=0.906, VAR2=0.606: {predicted_cluster[0]}")
