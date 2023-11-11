import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import string



# Load the trained Naive Bayes model
model = joblib.load('model.joblib')

# Load the CountVectorizer used for feature extraction during training
vectorizer = joblib.load('ve.joblib')


# Define a custom preprocessing function
def preprocess(input_text):
    # Remove punctuation and convert to lowercase

    input_text = input_text.translate(str.maketrans('', '', string.punctuation)).lower()

    # Tokenize by splitting on spaces
    words = input_text.split()

    # Remove stopwords (you can customize this list)
    stopwords = set(["the", "and", "a", "an", "in", "of", "to", "for", "it", "with"])
    words = [word for word in words if word not in stopwords]

    # Rejoin the words to form preprocessed text
    preprocessed_text = ' '.join(words)

    return preprocessed_text


# Function to perform sentiment analysis on input text
def perform_sentiment_analysis(input_text):
    # Preprocess the input text
    preprocessed_text = preprocess(input_text)

    # Use the loaded CountVectorizer to transform the preprocessed text into numerical features
    features = vectorizer.transform([preprocessed_text])

    # Use the trained Naive Bayes model to predict the sentiment
    sentiment_label = model.predict(features)[0]

    # Return the sentiment label (e.g., 'positive', 'negative', 'neutral')
    return sentiment_label


# Example input text
input_text = "today is very good day until i killed"

# Perform sentiment analysis
predicted_sentiment = perform_sentiment_analysis(input_text)

# Print the predicted sentiment
print(f"Predicted Sentiment: {predicted_sentiment}")
