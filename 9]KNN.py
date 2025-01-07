from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Using 3 nearest neighbors
knn.fit(X_train, y_train)

# Step 3: Prediction on the test data
y_pred = knn.predict(X_test)

# Step 4: Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Predicted labels:", y_pred)
print("Actual labels:", y_test)
print("Model Accuracy:", accuracy)
