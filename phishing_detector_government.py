import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load updated dataset
df = pd.read_csv("government_emails.csv")

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email_body"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Predict function
def check_email(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "Phishing Email" if result == 1 else "Legitimate Email"

# Example test
sample = "This is a reminder that your NIDA ID card is ready for collection."
print("Sample email analysis result:", check_email(sample))
