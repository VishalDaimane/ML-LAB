import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Check for null values
print("\nNull values:")
print(df.isnull().sum())

# Visualize data
df['MedHouseVal'].hist(bins=50, color='blue')
plt.title("Median House Value Distribution")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Covariance and correlation
print("\nCovariance:")
print(df.cov())
print("\nCorrelation:")
print(df.corr())

# Gradient Descent Regression
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class LinearRegressionGD:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters

    def fit(self, X, y):
        self.W = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.iters):
            y_pred = np.dot(X, self.W) + self.b
            dW = -(2 / len(y)) * np.dot(X.T, (y - y_pred))
            db = -(2 / len(y)) * np.sum(y - y_pred)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.W) + self.b

model = LinearRegressionGD(lr=0.01, iters=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMSE:", mse)
print("R2 Score:", r2)

plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
