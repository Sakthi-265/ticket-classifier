import os
import re
import string

import joblib
import spacy
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

app = Flask(__name__, static_folder=None)

artifacts = joblib.load(os.path.join(BASE_DIR, "ticket_model.joblib"))
count_vect = artifacts["count_vect"]
tfidf_transformer = artifacts["tfidf_transformer"]
model = artifacts["model"]

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = set(nlp.Defaults.stop_words)

MIN_LEN, MAX_LEN = 10, 3000


def clean_text(text):
    text = text.lower()
    text = re.sub("/[.*/]", "", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub("/S*/d/S*/s*", "", text).strip()
    return text.strip()


def preprocess(text):
    doc = nlp(clean_text(text))
    lemma = " ".join(t.lemma_ for t in doc if t.text not in STOPWORDS)
    return lemma.replace("xxxx", "")


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if len(text) < MIN_LEN:
        return jsonify(error=f"Text must be at least {MIN_LEN} characters."), 400
    if len(text) > MAX_LEN:
        return jsonify(error=f"Text must be at most {MAX_LEN} characters."), 400

    features = tfidf_transformer.transform(count_vect.transform([preprocess(text)]))
    probs = model.predict_proba(features)[0]
    idx = int(probs.argmax())
    return jsonify(category=str(model.classes_[idx]), confidence=round(float(probs[idx]), 4))


@app.route("/api/health")
def health():
    return jsonify(status="ok")


# Serve the frontend files from the parent folder
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)