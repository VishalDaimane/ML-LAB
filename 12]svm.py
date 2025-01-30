import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset from CSV file
# Replace 'your_dataset.csv' with the path to your CSV file
df = pd.read_csv('your_dataset.csv')

# Assuming the last column is the target variable and the rest are features
X = df.iloc[:, :-1]  # Features (all columns except the last one)
y = df.iloc[:, -1]   # Target (the last column)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
