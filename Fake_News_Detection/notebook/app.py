import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# NLP setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\\S+', '', text)

    # Remove numbers
    text = re.sub(r'\\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenize
    words = text.split()

    # Remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# UI
st.title("Fake News Detection App")

st.write("Enter a news article below:")

news = st.text_area("News Text")

if st.button("Predict"):

    # Preprocess input
    cleaned_news = preprocess_text(news)

    # Convert to vector
    vectorized_news = vectorizer.transform([cleaned_news])

    # Predict
    prediction = model.predict(vectorized_news)

    # Output
    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")