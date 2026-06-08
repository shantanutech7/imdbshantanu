
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded for the app environment
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Initialize lemmatizer and stopwords (must be done after download)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the saved model and vectorizer
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Make sure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()

# Preprocessing function (re-using the one from the notebook)
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app layout
st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to get its sentiment (positive/negative).")

user_input = st.text_area("Movie Review:", "Type your review here...")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)

        # Vectorize the preprocessed text
        input_vectorized = vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(input_vectorized)

        # Display result
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        st.success(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")

