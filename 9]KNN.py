import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV file
file_path = 'iris.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Assuming the last column is the target variable and the rest are features
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values   # Target variable (last column)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Simple 2D scatter plot using two features (e.g., sepal_length and sepal_width)
plt.figure(figsize=(8, 6))

# Plot the test data points, colored by their true labels
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='True Labels', edgecolor='k', s=100)

# Plot the test data points, colored by their predicted labels (for comparison)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='x', label='Predicted Labels', s=100)

# Add labels and title
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('True vs Predicted Labels (Sepal Length vs Sepal Width)')
plt.legend()
plt.show()
