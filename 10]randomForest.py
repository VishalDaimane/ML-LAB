import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset from CSV
file_path = "your_file.csv"  # Change this to your actual CSV file path
data = pd.read_csv(file_path)

# Assuming the last column is the target variable
y = data.iloc[:, -1]  # Target variable
X = data.iloc[:, :-1]  # Features

# Display the first 5 rows of the features and target
print("Features:\n", X.head())
print("Target:\n", y.head())

# 3. Visualize data
data.hist(figsize=(10, 8))
plt.show()

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)
