import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import  joblib
# Step 1: Load and preprocess the dataset
data = pd.read_csv('Tweets_2.csv')

data['selected_text'].fillna('', inplace=True)  # Replace NaN with an empty string

text = data['selected_text'].values
labels = data['sentiment'].values
# Step 2: Convert text data into numerical feature vectors
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(text)

# Step 3: Train the Naive Bayes model
nb = MultinomialNB()
nb.fit(features, labels)

# Step 4: Predict sentiment on new data
new_text = ["I love this movie!", "This product is terrible.", "The food was delicious."]
new_features = vectorizer.transform(new_text)
new_predictions = nb.predict(new_features)
print(new_predictions)

# Step 5: Generate the classification report to evaluate the model
predictions = nb.predict(features)
print(classification_report(labels, predictions))

print(accuracy_score(labels, predictions))

joblib.dump(nb,'model.joblib')
joblib.dump(vectorizer,'ve.joblib')