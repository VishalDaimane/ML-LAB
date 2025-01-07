import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the dataset
df = fetch_california_housing(as_frame=True).frame

# Display first 5 rows
print(df.head())

# Visualizations
sns.histplot(df['MedHouseVal'], bins=30, kde=True)
plt.title("House Value Distribution")
plt.show()

# Correlation and covariance
print(df.cov(), df.corr())

# Train-test split
X = df.drop('MedHouseVal', axis=1)  
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and predict
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
