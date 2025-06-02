# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample data: Customer reviews (you can replace this with your own dataset)
data = {
    'review': [
        "I love this product! It's amazing and works wonderfully.",
        "Terrible experience, the item broke after one use.",
        "Very happy with my purchase. Great quality and fast delivery.",
        "I hate it. Completely useless and a waste of money.",
        "The product is okay, not the best but decent.",
        "Excellent value for money. Highly recommend it!",
        "Not good. I expected better performance.",
        "Fantastic! I will buy this again for sure.",
        "Worst purchase ever, do not buy this.",
        "Satisfied but could be improved."
    ],
    'sentiment': [
        1,  # positive
        0,  # negative
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# View sample data
print("Sample data:")
print(df.head())

# Split data into features and target
X = df['review']
y = df['sentiment']

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_vec, y_train)

# Predict sentiment on test data
y_pred = model.predict(X_test_vec)

# Evaluate Model
print("\nAccuracy on test data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
