import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

# Display first 5 rows
print(housing_df.head())

# Check for null values
print("\nNull values:\n", housing_df.isnull().sum())

# Visualize data
sns.pairplot(housing_df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']])
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(housing_df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Split data
X = housing_df.drop(columns=['MedHouseVal'])
y = housing_df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("R^2 (Accuracy):", r2)
print(f"Accuracy: {r2 * 100:.2f}%")


# Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
