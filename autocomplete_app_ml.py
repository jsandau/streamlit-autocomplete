import streamlit as st
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
import numpy as np

nltk.download('all')

st.title("ML Autocomplete")

# Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8").lower()
    tokens = word_tokenize(text)
    
    # Create bigrams
    bigrams = list(ngrams(tokens, 2))
    
    # Split into X (first word) and y (next word)
    X = [w1 for w1, w2 in bigrams]
    y = [w2 for w1, w2 in bigrams]

    # Vectorize the first word (X)
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X)

    # Train Naive Bayes
    model = MultinomialNB()
    model.fit(X_vect, y)

    # Input box
    user_input = st.text_input("Type something...")

    if user_input.endswith(" "):
        last_word = user_input.strip().split()[-1]
        last_word_vect = vectorizer.transform([last_word])
        
        if last_word_vect.nnz == 0:
            st.write("No predictions found (word not in training set).")
        else:
            probs = model.predict_proba(last_word_vect)[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            suggestions = np.array(model.classes_)[top_indices]

            st.write("Next word suggestions:")
            for s in suggestions:
                st.write(f"→ {user_input}{s}")
