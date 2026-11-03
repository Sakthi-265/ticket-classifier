import json
import logging
import os
import re
import string

import azure.functions as func
import joblib
import spacy

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ---------- Load artifacts once (cold start only) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
artifacts = joblib.load(os.path.join(BASE_DIR, "ticket_model.joblib"))
count_vect = artifacts["count_vect"]
tfidf_transformer = artifacts["tfidf_transformer"]
model = artifacts["model"]

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = set(nlp.Defaults.stop_words)

MAX_LEN = 3000
MIN_LEN = 10


# ---------- Preprocessing (must mirror the notebook exactly) ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub("/[.*/]", "", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub("/S*/d/S*/s*", "", text).strip()
    return text.strip()


def lemmatize(text: str) -> str:
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc if tok.text not in STOPWORDS)


def preprocess(text: str) -> str:
    lemma = lemmatize(clean_text(text))
    return lemma.replace("xxxx", "")


def json_response(body: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body),
        status_code=status,
        mimetype="application/json",
    )


# ---------- Routes ----------
@app.route(route="classify", methods=["POST"])
def classify(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return json_response({"error": "Request body must be valid JSON."}, 400)

    text = (payload.get("text") or "").strip() if isinstance(payload, dict) else ""
    if len(text) < MIN_LEN:
        return json_response({"error": f"Text must be at least {MIN_LEN} characters."}, 400)
    if len(text) > MAX_LEN:
        return json_response({"error": f"Text must be at most {MAX_LEN} characters."}, 400)

    try:
        processed = preprocess(text)
        features = tfidf_transformer.transform(count_vect.transform([processed]))
        probs = model.predict_proba(features)[0]
        idx = int(probs.argmax())
        return json_response(
            {
                "category": str(model.classes_[idx]),
                "confidence": round(float(probs[idx]), 4),
            }
        )
    except Exception:
        logging.exception("Classification failed")
        return json_response({"error": "Internal error while classifying."}, 500)


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    return json_response({"status": "ok"})