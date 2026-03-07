# 📰 Fake News Detector

A simple Machine Learning application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP).

The model processes news text, cleans it using NLP techniques, and predicts authenticity using a trained classification model.

---

## ⚙️ Features

- Text preprocessing and cleaning
- Stopword removal and stemming using **Porter Stemmer**
- Feature extraction using **CountVectorizer**
- Classification using **Naive Bayes**
- Fast prediction for news text input

---

## 🧠 ML Pipeline

1. Text Cleaning using **regex**
2. Stopword removal with **NLTK corpus**
3. Word stemming using **PorterStemmer**
4. Text vectorization using **CountVectorizer**
5. Model training using **Naive Bayes classifier**

---

## 🛠️ Tech Stack

- Python
- NLTK
- Scikit-learn
- Pandas
- NumPy
- Streamlit (for UI)

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py