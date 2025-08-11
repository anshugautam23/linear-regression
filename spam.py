import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset
data = {
    "message": [
        "Win a free iPhone now!!! Click here to claim your prize.",
        "Hey, are we still meeting for lunch?",
        "Congratulations! You have won a lottery worth $1,000,000!",
        "Don't forget to submit your assignment by tonight.",
        "URGENT: Your bank account has been compromised. Login immediately."
    ],
    "label": ["spam", "ham", "spam", "ham", "spam"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()
    return text

df["cleaned_message"] = df["message"].apply(clean_text)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_message"], df["label"], test_size=0.2, random_state=42)

# Text Vectorization & Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train Model
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test with New Message
new_messages = ["Claim your free vacation now!", "Hey, can you send me the report?"]
predictions = model.predict(new_messages)
print("\nPredictions:", predictions)
