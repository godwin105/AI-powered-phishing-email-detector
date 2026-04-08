import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the updated dataset
df = pd.read_csv("government_emails.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email_body"])
y = df["label"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Government Phishing Detector")
st.title("Tanzania AI-Powered Government Email Phishing Detector")

st.markdown("Analyze if a Tanzanian government-related email is legitimate or phishing.")

email_input = st.text_area("📩 Paste the email content below:")

if st.button("Analyze Email"):
    vec = vectorizer.transform([email_input])
    prediction = model.predict(vec)[0]
    result = "🚨 Phishing Email Detected!" if prediction == 1 else "✅ Legitimate Government Email"

    st.subheader("📊 Result:")
    st.success(result if prediction == 0 else "")
    st.error(result if prediction == 1 else "")

def main():
    import subprocess
    subprocess.run(["streamlit", "run", __file__])

if __name__ == "__main__":
    main()
