"""
AI Scam Detector — backend

Trains/loads a Random Forest classifier over 10 keyword-based feature
categories (urgency, money request, official appearance, etc.) and serves
both the JSON prediction API and the landing page.

Run:
    pip install -r requirements.txt
    python app.py
"""

import os
import csv

import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(APP_DIR, "labeled_dataset.csv")
MODEL_PATH = os.path.join(APP_DIR, "scam_detector_model.pkl")

app = Flask(__name__)
model = None

# ---------------------------------------------------------------------------
# Feature categories
#
# Each category is a (label, keyword list) pair. A message's score for a
# category is the fraction of that category's keywords found in it — the
# same scheme the original scam_detector.py / endpoints.py used. Keeping the
# keyword lists (rather than just the scores) also lets /predict report
# *which* words in a given message actually tripped each category, so the
# UI can explain a verdict instead of just stating one.
# ---------------------------------------------------------------------------

CATEGORIES = {
    "urgency": (
        "Urgency",
        [
            "urgent", "immediately", "asap", "now", "instantly", "right away", "critical",
            "emergency", "act fast", "without delay", "rush", "time sensitive", "immediate attention",
            "important", "priority", "respond quickly", "final notice", "quickly", "within hours", "last chance",
        ],
    ),
    "money_request": (
        "Money request",
        [
            "send money", "payment", "bank account", "transfer", "fund", "financial assistance",
            "deposit", "remit", "wire", "moneygram", "western union", "btc", "crypto", "currency",
            "dollars", "cash", "fee", "transaction", "cheque", "inheritance", "unclaimed funds", "reward",
            "$", "million", "billion", "usd", "50m", "50 million", "50$", "50000", "lot of money", "wealth",
        ],
    ),
    "official_appearance": (
        "Official appearance",
        [
            "official", "government", "irs", "fbi", "customs", "account update", "verification",
            "authority", "compliance", "legal", "investigation", "officer", "department", "administrator",
            "state", "national", "hq", "regulation", "policy", "internal audit",
        ],
    ),
    "reward_offer": (
        "Reward offer",
        [
            "win", "congratulations", "lucky", "jackpot", "lottery", "cash prize", "gift card",
            "you've won", "sweepstakes", "bingo", "claim prize", "million", "billion", "bonanza",
            "reward", "exclusive prize", "you qualify", "redeem", "receive funds", "special winner",
        ],
    ),
    "celebrity_reference": (
        "Celebrity reference",
        [
            "elon musk", "taylor swift", "jeff bezos", "bill gates", "oprah", "lebron", "cristiano",
            "selena", "beyonce", "trump", "biden", "modi", "virat", "shahrukh", "kardashian",
            "celebrity", "hollywood", "influencer", "verified", "blue tick",
        ],
    ),
    "grammar_issues": (
        "Grammar issues",
        [
            "recieve", "seperated", "definately", "adress", "freind", "untill", "wich", "immediatly",
            "inconvienent", "completly", "alot", "happend", "beleive", "enviroment", "goverment",
            "neccessary", "occurence", "seperate", "succesful", "truely",
        ],
    ),
    "unusual_contact_method": (
        "Unusual contact method",
        [
            "telegram", "whatsapp", "sms", "text", "chat", "dm", "message me", "contact via app",
            "reach me", "wechat", "imo", "viber", "signal", "messenger", "snapchat", "facebook",
            "call this number", "ping me", "alternative number", "line",
        ],
    ),
    "pressure_to_act": (
        "Pressure to act",
        [
            "act now", "limited time", "only today", "last chance", "hurry", "urgent deadline",
            "before it's too late", "don't miss out", "one-time offer", "expires soon",
            "final offer", "time running out", "claim fast", "do not delay", "fast response",
            "limited stock", "urgent response needed", "need quick answer", "instantly confirm", "must act quickly",
        ],
    ),
    "suspicious_link": (
        "Suspicious link",
        [
            "http", "https", "bit.ly", "tinyurl", "shorturl", "redirect", ".xyz", ".top", ".win",
            "click here", "open link", "see details", "login page", "promo code", "verify link",
            "security page", "unusual login", "confirm access", "track order", "claim voucher",
        ],
    ),
    "upfront_payment": (
        "Upfront payment",
        [
            "pay upfront", "advance payment", "initial deposit", "send fee", "registration fee",
            "processing charge", "application cost", "service fee", "transfer cost", "one-time charge",
            "security fee", "membership fee", "setup cost", "handling fee", "deposit first",
            "pay before", "cash advance", "shipping fee", "booking charge", "consultation fee",
        ],
    ),
}

# Fixed order — must match the column order labeled_dataset.csv was built with.
CATEGORY_ORDER = [
    "urgency", "money_request", "official_appearance", "reward_offer", "celebrity_reference",
    "grammar_issues", "unusual_contact_method", "pressure_to_act", "suspicious_link", "upfront_payment",
]


def analyze_message(message: str):
    """Score a message against every category and report which keywords matched."""
    message_lower = message.lower()
    breakdown = []
    for key in CATEGORY_ORDER:
        label, keywords = CATEGORIES[key]
        found = [kw for kw in keywords if kw in message_lower]
        score = round(len(found) / len(keywords), 2)
        breakdown.append({"key": key, "label": label, "score": score, "matches": found})
    feature_vector = [item["score"] for item in breakdown]
    return feature_vector, breakdown


def load_and_train_model():
    """Load a cached model if present, otherwise train one from the dataset."""
    global model

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Loaded existing model from scam_detector_model.pkl")
            return
        except Exception:
            print("Failed to load existing model, training a new one")

    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # header
                rows = [row for row in reader if row]

            messages = [row[0] for row in rows]
            labels = [1 if row[1].lower() == "scam" else 0 for row in rows]
            features = [analyze_message(m)[0] for m in messages]

            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_PATH)

            predictions = model.predict(X_test)
            accuracy = sum(p == y for p, y in zip(predictions, y_test)) / len(y_test)
            print(f"Trained new model — test accuracy: {accuracy:.2%}")
        except Exception as e:
            print(f"Error training model ({e}); falling back to a minimal dummy model")
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit([[0.0] * len(CATEGORY_ORDER), [0.5] * len(CATEGORY_ORDER)], [0, 1])
    else:
        print("No labeled_dataset.csv found; using a minimal dummy model")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit([[0.0] * len(CATEGORY_ORDER), [0.5] * len(CATEGORY_ORDER)], [0, 1])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify({"error": "No message provided"}), 400
        if model is None:
            return jsonify({"error": "Model not available"}), 500

        feature_vector, breakdown = analyze_message(message)
        prediction = model.predict([feature_vector])[0]
        proba = model.predict_proba([feature_vector])[0]

        # classes_ is sorted ascending ([0, 1]) for binary 0/1 labels, so
        # index 1 is always the probability of the "scam" class regardless
        # of which class was predicted.
        scam_index = list(model.classes_).index(1)
        risk_score = round(float(proba[scam_index]) * 100, 1)

        return jsonify(
            {
                "prediction": "SCAM" if prediction == 1 else "NOT SCAM",
                "risk_score": risk_score,
                "confidence": round(float(max(proba)) * 100, 1),
                "message_length": len(message),
                "categories": sorted(breakdown, key=lambda c: c["score"], reverse=True),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Error processing request: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    print("Starting Scam Detector...")
    load_and_train_model()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"Serving on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
