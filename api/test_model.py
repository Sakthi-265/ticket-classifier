import re
import string

import joblib
import spacy

artifacts = joblib.load("ticket_model.joblib")
count_vect = artifacts["count_vect"]
tfidf_transformer = artifacts["tfidf_transformer"]
model = artifacts["model"]

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = set(nlp.Defaults.stop_words)


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


def classify(text):
    features = tfidf_transformer.transform(count_vect.transform([preprocess(text)]))
    probs = model.predict_proba(features)[0]
    idx = probs.argmax()
    return model.classes_[idx], round(float(probs[idx]), 4)


tests = [
    "I was charged an overdraft fee even though I had enough funds in my checking account.",
    "There is an unauthorized charge on my credit card and the merchant refuses to refund me.",
    "My mortgage lender is charging me incorrect monthly mortgage payments.",
]

for t in tests:
    print(classify(t), "<-", t)