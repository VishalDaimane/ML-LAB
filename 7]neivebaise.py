from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Dataset
texts = [
    "I love this sandwich", "This is an amazing place", 
    "I feel very good about these cheese", "This is my best work", 
    "What an awesome view", "I do not like this restaurant", 
    "I am tired of this stuff", "I can’t deal with this", 
    "He is my sworn enemy", "My boss is horrible", 
    "This is an awesome place", "I do not like the taste of this juice", 
    "I love to dance", "I am sick and tired of this place", 
    "What a great holiday", "That is a bad locality to stay", 
    "We will have good fun tomorrow", "I went to my enemys house today"
]
labels = ["pos", "pos", "pos", "pos", "pos", "neg", "neg", "neg", 
          "neg", "neg", "pos", "neg", "pos", "neg", "pos", "neg", 
          "pos", "neg"]

# Vectorize text
X = CountVectorizer().fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train and predict
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, pos_label="pos"))
print("Precision:", precision_score(y_test, y_pred, pos_label="pos"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["pos", "neg"]))
