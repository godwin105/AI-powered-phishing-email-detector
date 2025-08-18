# 🇹🇿 AI-Powered Government Email Phishing Detection

This project demonstrates a machine learning-based system that detects phishing emails, specifically tailored to government users in Tanzania. It uses natural language processing (NLP) and logistic regression to classify emails as **legitimate** or **phishing**.

## 📁 Files Included

- `government_emails.csv` — A dataset of 20 realistic emails (10 phishing, 10 legitimate), based on Tanzanian government communication patterns.
- `phishing_detector_government.py` — Python script to train and test the phishing detection model.
- `phishing_streamlit_government.py` — A Streamlit web app interface to test email messages interactively.

## 🧪 How to Run (Terminal Version)

1. Install Python packages:
   ```bash
   pip install pandas scikit-learn
   ```

2. Run the script:
   ```bash
   python phishing_detector_government.py
   ```

3. You will see:
   - A classification report showing the model’s performance.
   - An example email classification result.

## 🌐 How to Run the Web App (Streamlit)

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```

2. Launch the web app:
   ```bash
   streamlit run phishing_streamlit_government.py
   ```

3. Paste any email message into the box and press **"Analyze Email"** to classify it.

## 🛡 Use Case

This project aligns with the **NIST Cybersecurity Framework** and **Tanzania’s Data Protection Act (2022)** by providing:

- Real-time phishing detection
- Data privacy (no emails leave the system)
- Secure classification logic with local relevance

## ✅ Sample Legitimate Emails

- “Please review the attached document for the ministry's Q2 financial report.”
- “The cabinet meeting will be held on Monday at 10:00 AM.”

## 🚨 Sample Phishing Emails

- “Your NIDA account is about to expire. Click here to verify your details.”
- “Immediate action required: Reset your Ministry of Health login credentials.”

---

🔒 Built for academic and cybersecurity awareness purposes.
