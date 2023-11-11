# myapp/views.py

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .forms import TextInputForm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import joblib
import string
import re

def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text

# Load the trained Naive Bayes model
model1 = joblib.load('mlmodel/model.joblib')


@csrf_exempt
def analysis(request):
    form = TextInputForm(request.POST or None)
    sentiment = None
    res = None
    if form.is_valid():
        text = form.cleaned_data['text']
        # Preprocess and vectorize the text (similar to your training data preprocessing)
        preprocessed_text = preprocess(text)
        vectorizer = joblib.load('mlmodel/ve.joblib')
        new_text = [text]
        new_features = vectorizer.transform(new_text)
        sentiment = model1.predict(new_features)

        res=model1.predict_proba(new_features)

    return render(request, 'First/index.html', {'form': form, 'sentiment': sentiment,'res':res})

def home(request):
    return render(request, 'First/exp.html')
