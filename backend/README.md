# AI Scam Detector

A Random Forest classifier over 10 keyword-based signals (urgency, money
requests, official-looking language, reward offers, and so on) that scores
a message's likelihood of being a scam — and explains exactly which words
tripped which signal.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Then open http://localhost:5000. The first run trains the model from
`labeled_dataset.csv` and caches it to `scam_detector_model.pkl`; later
runs load the cached model instantly.

## What's here

- `app.py` — Flask app: feature extraction, model training/loading, and
  two routes: `POST /predict` (returns prediction, risk score, and a
  per-category breakdown of matched keywords) and `GET /health`.
- `templates/index.html` — the landing page. The hero visual at the top is
  a scripted animation for atmosphere; the **"Try it on a real message"**
  section further down calls `/predict` for real and renders the actual
  model output, including highlighting the exact phrases that matched.
- `labeled_dataset.csv` — training data.

## API

```
POST /predict
{ "message": "Congratulations! You've won $5,000! Claim now..." }
```

```json
{
  "prediction": "SCAM",
  "risk_score": 87.0,
  "confidence": 87.0,
  "message_length": 118,
  "categories": [
    { "key": "reward_offer", "label": "Reward offer", "score": 0.15, "matches": ["congratulations", "win"] }
  ]
}
```
