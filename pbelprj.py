import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#LOAD DATASET 

df = pd.read_csv("tweet_emotions.csv")

print("Dataset Preview:")
print(df.head())

print("\nEmotion Distribution:")
print(df["sentiment"].value_counts())


# TEXT CLEANING 

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)     # remove punctuation

    return text


df["clean_text"] = df["content"].apply(clean_text)


# VISUALIZATION

df["sentiment"].value_counts().plot(kind="bar")
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()

df["sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.title("Emotion Distribution")
plt.ylabel("")
plt.show()


#FEATURES

X = df["clean_text"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# TF-IDF

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# MODEL 1 

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

print("\nNaive Bayes Accuracy:")
print(accuracy_score(y_test, nb_pred))


#MODEL 2 

lr_model = LogisticRegression(max_iter=200)

lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))


#EVALUATION 

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))


#USER PREDICTION 

print("\n------ EMOTION DETECTOR ------")

while True:

    text = input("Enter a sentence (type exit): ")

    if text.lower() == "exit":
        break

    text = clean_text(text)

    vec = vectorizer.transform([text])

    prediction = lr_model.predict(vec)

    print("Predicted Emotion:", prediction[0])
