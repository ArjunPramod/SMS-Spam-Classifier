# 📩 SMS Spam Classifier

An end-to-end NLP project to classify SMS messages as **Spam** or **Ham (Not Spam)** using traditional machine learning, TF–IDF features, and a Streamlit web interface.

---

## 🚀 Overview

This project demonstrates a full ML workflow:

1. **Data ingestion** from Kaggle’s SMS Spam Collection Dataset  
2. **Text preprocessing** with NLTK (cleaning, stopword removal, stemming)  
3. **Feature extraction** using TF–IDF  
4. **Model training & evaluation** with Logistic Regression (and optional Naive Bayes)  
5. **Model persistence** with `joblib`  
6. **Interactive web app** built with Streamlit  
7. **Deployment-ready** for Streamlit Community Cloud  

---

## 📊 Dataset

- **Name:** SMS Spam Collection Dataset  
- **Source (Kaggle):** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
- **Instances:** ~5.5k SMS messages labeled as `ham` or `spam`  

Please refer to the dataset page for licensing and citation details.

---

## 🧰 Tech Stack

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn` (TF–IDF, Logistic Regression, Naive Bayes, metrics)
  - `nltk` (stopwords, stemming)
  - `joblib` (model serialization)
  - `streamlit` (web app)

---

## 📁 Project Structure

```bash
sms-spam-classifier/
├── app.py                     # Streamlit app
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data/
│   └── spam.csv               # Kaggle dataset (placed here by you)
├── models/
│   ├── spam_model.pkl         # Trained Logistic Regression model
│   └── vectorizer.pkl         # TF–IDF vectorizer
└── notebooks/
    └── sms_spam_classifier.ipynb  # Training & evaluation notebook
