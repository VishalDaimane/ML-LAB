
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.linspace(1, 10, 100)  # 100 data points between 1 and 10
y = 2 * np.sin(X) + np.log(X) + np.random.normal(0, 0.2, 100)  # Non-linear data

# Locally Weighted Regression function
def locally_weighted_regression(X, y, query_point, c):
    m = len(X)
    X_bias = np.c_[np.ones(m), X]  # Add bias term to X
    query_bias = np.array([1, query_point])  # Bias for the query point
    
    # Calculate weights
    weights = np.exp(-((X - query_point) ** 2) / (2 * c ** 2))
    W = np.diag(weights)  # Create diagonal weight matrix
    
    # Compute beta
    beta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)
    
    # Predict Y for the query point
    prediction = query_bias @ beta
    return prediction

# Predict over the data range
c = 0.8  # Bandwidth parameter
y_pred = np.array([locally_weighted_regression(X, y, x, c) for x in X])

# Plot original data and LWR fit
plt.scatter(X, y, color='blue', label='Original Data', alpha=0.6)
plt.plot(X, y_pred, color='red', label='LWR Fit', linewidth=2)
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
