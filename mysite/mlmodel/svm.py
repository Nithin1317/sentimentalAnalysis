from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,  ConfusionMatrixDisplay
import pandas as pd

import joblib

data=pd.read_csv("Tweets_2.csv")
X = data['selected_text'].values.astype('U')
y = data['sentiment'].values.astype('U')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

# Create TF-IDF vectors from the text data
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

pipe = Pipeline([('tfidf_vectorizer', TfidfVectorizer(lowercase=True,
                                                      stop_words='english',
                                                      analyzer='word')),

                 ('naive_bayes', MultinomialNB())])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

pipe.fit(list(X_train), list(y_train))
y_pred = pipe.predict(X_test)
print(confusion_matrix(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
pipe['naive_bayes']
# Make predictions on the test set

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
#
joblib.dump(pipe,'svm_model.joblib')
joblib.dump(vectorizer,'svm_ve.joblib')

