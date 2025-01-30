from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Import dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Display first 5 rows
print(df.head())

# 3. Check for null values
print(df.isnull().sum())

# 4. Visualize data
df.plot.scatter(x='MedInc', y='MedHouseVal')

# 5. Covariance and correlation
print("Covariance:\n", df.cov())
print("Correlation:\n", df.corr())

# 6. Train-test split
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Gradient Descent
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
