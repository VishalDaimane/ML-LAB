import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

# 1. Import dataset
data = load_iris()
iris = pd.DataFrame(data.data, columns=data.feature_names)
iris['species'] = data.target

# Map target numbers to species names
species_mapping = {i: name for i, name in enumerate(data.target_names)}
iris['species'] = iris['species'].map(species_mapping)

# 2. Display first 5 rows
print("First 5 rows:")
print(iris.head())

# 3. Check the number of samples of each class
print("\nSamples per class:")
print(iris['species'].value_counts())

# 4. Check for null values
print("\nNull values:")
print(iris.isnull().sum())

# Encode species column for numerical operations
iris['species'] = LabelEncoder().fit_transform(iris['species'])

# 5. Visualize the data in the form of graphs
sns.pairplot(iris, hue='species', diag_kind='kde', palette='husl')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# 6. Obtain covariance and correlation values
print("\nCovariance matrix:")
print(iris.cov())

print("\nCorrelation matrix:")
print(iris.corr())

# 6. Train and test model
X = iris.iloc[:, :-1]
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
