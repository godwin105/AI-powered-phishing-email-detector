import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

TEST_EMAILS = [
    ("This is a reminder that your NIDA ID card is ready for collection.", 0),
    ("The cabinet meeting will be held on Monday at 10:00 AM.", 0),
    ("Taarifa ya mwaka ya wizara ya fedha imetolewa leo kwa ajili ya mapitio.", 0),
    ("Fomu za tathmini ya utendaji kazi zinapatikana sasa katika mfumo wa HRMS.", 0),
    ("URGENT: Your government email will be suspended. Click here to verify.", 1),
    ("TRA tax refund of TSH 2,500,000 pending. Submit bank details to receive it.", 1),
    ("Akaunti yako ya NIDA itafungwa kama hautathibitisha taarifa zako.", 1),
    ("Umepata zawadi ya TSH 5,000,000 kutoka Ofisi ya Rais. Bonyeza hapa kudai.", 1),
]


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])


def evaluate(df):
    X, y = df["email_body"], df["label"]

    # ── Hold-out evaluation ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("=" * 50)
    print("HOLD-OUT EVALUATION  (75% train / 25% test)")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # ── Cross-validation ─────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe_cv = build_pipeline()
    acc = cross_val_score(pipe_cv, X, y, cv=cv, scoring="accuracy")
    f1  = cross_val_score(pipe_cv, X, y, cv=cv, scoring="f1")

    print()
    print("=" * 50)
    print("5-FOLD CROSS-VALIDATION  (full dataset)")
    print("=" * 50)
    print(f"Accuracy : {acc.mean():.3f} ± {acc.std():.3f}  {[f'{s:.3f}' for s in acc]}")
    print(f"F1-Score : {f1.mean():.3f} ± {f1.std():.3f}  {[f'{s:.3f}' for s in f1]}")

    # ── Final model on all data ───────────────────────────────────────────
    final = build_pipeline()
    final.fit(X, y)
    return final


def check_email(model, text):
    proba = model.predict_proba([text])[0]
    pred  = model.predict([text])[0]
    label = "Phishing Email" if pred == 1 else "Legitimate Email"
    confidence = max(proba) * 100
    return label, confidence


if __name__ == "__main__":
    print("Loading dataset and training model...\n")
    df = pd.read_csv("government_emails.csv")
    print(f"Dataset: {len(df)} emails  |  "
          f"Phishing: {df['label'].sum()}  |  "
          f"Legitimate: {(df['label'] == 0).sum()}\n")

    model = evaluate(df)

    print()
    print("=" * 50)
    print("SAMPLE EMAIL PREDICTIONS")
    print("=" * 50)
    for text, true_label in TEST_EMAILS:
        label, conf = check_email(model, text)
        expected = "Phishing Email" if true_label == 1 else "Legitimate Email"
        status = "OK" if label == expected else "FAIL"
        print(f"\n  [{status}] [{label:<18}] ({conf:.1f}%)")
        print(f"    Email: {text[:80]}{'...' if len(text) > 80 else ''}")
