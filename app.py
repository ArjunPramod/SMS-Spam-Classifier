import os
import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """
    Same preprocessing as in the training notebook:
    - lowercase
    - remove non-letters
    - remove stopwords
    - stemming
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "spam_model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_artifacts()

label_inv_map = {0: "Not Spam (Ham)", 1: "Spam"}

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.write(
    "Type or paste an SMS message below and click **Classify** to see whether "
    "it is predicted as spam or not."
)

user_input = st.text_area("SMS Message", height=150, placeholder="Enter your SMS text here...")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message before classifying.")
    else:
        processed = preprocess_text(user_input)
        vec = vectorizer.transform([processed])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0, 1]  # probability of spam

        result_label = label_inv_map[prediction]
        st.subheader("Prediction")
        if prediction == 1:
            st.error(f"🔴 {result_label}")
        else:
            st.success(f"🟢 {result_label}")

        st.write(f"**Estimated spam probability:** `{proba:.4f}`")
