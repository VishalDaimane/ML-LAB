from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# 1. Import dataset from CSV file
csv_path = "path/to/your/csv/file.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_path)

# 2. Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# 3. Check number of samples for each class
print("\nNumber of samples for each class:")
print(data['Species'].value_counts())

# 4. Check for null values
print("\nNull values in the dataset:")
print(data.isnull().sum())

# 5. Visualize data (bar plot of class distribution using pandas)
data['Species'].value_counts().plot(kind='bar', title="Class Distribution")

# 6. Covariance and correlation
print("\nCovariance Matrix:")
print(data.cov())
print("\nCorrelation Matrix:")
print(data.corr())

# 7. Train-test split
X = data.drop('Species', axis=1)  # Features
y = data['Species']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train model
model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
