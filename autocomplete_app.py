import streamlit as st
import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter

nltk.download('punkt')

st.title("Autocomplete from Uploaded Text")

# Upload file
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

# Only run if a file is uploaded
if uploaded_file:
    text = uploaded_file.read().decode("utf-8").lower()

    # Tokenize and build bigram model
    tokens = nltk.word_tokenize(text)
    bigrams = list(ngrams(tokens, 2))
    model = defaultdict(Counter)
    for w1, w2 in bigrams:
        model[w1][w2] += 1

    def predict_next_word(word, top_k=3):
        return model[word].most_common(top_k)

    # User input box
    user_input = st.text_input("Type something and press Enter:")

    if user_input:
        last_word = user_input.strip().split()[-1]
        suggestions = predict_next_word(last_word)
        if suggestions:
            st.write("Next word suggestions:")
            for word, _ in suggestions:
                st.write(f"→ {user_input} {word}")
        else:
            st.write("No suggestions found.")
else:
    st.write("Please upload a .txt file to get started.")
