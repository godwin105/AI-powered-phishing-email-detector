import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

SUSPICIOUS_KEYWORDS = [
    "click here", "verify", "expire", "urgent", "immediate", "suspended",
    "disabled", "reset password", "credentials", "confirm", "bank details",
    "claim", "congratulations", "selected", "alert", "warning", "breach",
    "bonyeza", "thibitisha", "haraka", "dharura", "fungua", "nywila",
    "kiungo", "malipo", "zawadi", "hongera", "akaunti itafungwa",
]

PHISHING_EXAMPLES = [
    "URGENT: Your government email will be suspended in 24 hours. Click here to verify your credentials immediately.",
    "Your NIDA account is about to expire. Click here to verify your details.",
    "TRA tax refund of TSH 2,500,000 is pending. Submit your bank details to receive it.",
    "Akaunti yako ya NIDA itafungwa kama hautathibitisha taarifa zako.",
]

LEGIT_EXAMPLES = [
    "The cabinet meeting will be held on Monday at 10:00 AM in the main conference hall.",
    "TRA will be conducting a taxpayer education seminar next week.",
    "Taarifa ya mwaka ya wizara ya fedha imetolewa leo kwa ajili ya mapitio.",
    "Staff are reminded to submit their annual asset declaration forms by end of quarter.",
]


@st.cache_resource(show_spinner="Training model on dataset...")
def load_model():
    df = pd.read_csv("government_emails.csv")
    X, y = df["email_body"], df["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])
    pipeline.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    return pipeline, cv_scores, len(df), int(y.sum()), int((y == 0).sum())


def detect_suspicious_keywords(text):
    text_lower = text.lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]


def main():
    st.set_page_config(
        page_title="Tanzania Phishing Detector",
        page_icon="🛡️",
        layout="centered",
    )

    st.title("🛡️ Tanzania AI-Powered Government Email Phishing Detector")
    st.markdown(
        "Paste any Tanzanian government-related email to classify it as "
        "**legitimate** or **phishing** — supports English and Swahili."
    )

    pipeline, cv_scores, total, phishing_count, legit_count = load_model()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Cross-Val Accuracy", f"{cv_scores.mean():.1%}")
        st.metric("Std Deviation", f"± {cv_scores.std():.1%}")
        st.divider()
        st.metric("Training Emails", total)
        st.metric("Phishing Samples", phishing_count)
        st.metric("Legitimate Samples", legit_count)
        st.divider()
        st.caption("Model: Logistic Regression + TF-IDF bigrams")
        st.caption("Languages: English + Swahili")
        st.caption("Agencies: NIDA · TRA · eGA · TCRA · PSC · MoH")

    # ── Input area ────────────────────────────────────────────────────────
    if "email_text" not in st.session_state:
        st.session_state.email_text = ""

    with st.expander("💡 Load a sample email"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Phishing samples**")
            for i, ex in enumerate(PHISHING_EXAMPLES):
                if st.button(f"Phishing #{i+1}", key=f"p{i}"):
                    st.session_state.email_text = ex
                    st.rerun()
        with col2:
            st.markdown("**Legitimate samples**")
            for i, ex in enumerate(LEGIT_EXAMPLES):
                if st.button(f"Legit #{i+1}", key=f"l{i}"):
                    st.session_state.email_text = ex
                    st.rerun()

    email_input = st.text_area(
        "📩 Paste the email content below:",
        value=st.session_state.email_text,
        height=180,
        placeholder="e.g. Your NIDA account will expire. Click here to verify...",
    )

    if st.button("🔍 Analyze Email", type="primary"):
        text = email_input.strip()
        if len(text) < 5:
            st.warning("Please enter a valid email message (at least 5 characters).")
        else:
            proba = pipeline.predict_proba([text])[0]
            prediction = pipeline.predict([text])[0]
            phishing_pct = proba[1] * 100
            legit_pct = proba[0] * 100

            st.divider()
            st.subheader("📊 Analysis Result")

            if prediction == 1:
                st.error(f"🚨 **Phishing Email Detected!** — Confidence: {phishing_pct:.1f}%")
            else:
                st.success(f"✅ **Legitimate Government Email** — Confidence: {legit_pct:.1f}%")

            col1, col2 = st.columns(2)
            col1.metric("Phishing probability", f"{phishing_pct:.1f}%")
            col2.metric("Legitimate probability", f"{legit_pct:.1f}%")

            st.progress(phishing_pct / 100, text="Phishing likelihood")

            found = detect_suspicious_keywords(text)
            if found:
                st.warning(f"⚠️ Suspicious signals detected: **{', '.join(found)}**")
            else:
                st.info("No known suspicious keywords detected in the text.")

            st.divider()
            st.caption("🔒 All analysis runs locally — no email content is sent externally.")


if __name__ == "__main__":
    main()
