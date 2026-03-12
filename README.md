# PBEL-Internship
NAME- Hiya Khichi
BATCH- 8
[PROJECT- AI BASED EMOTION RECOGNITION SYSTEM FROM TEXT]
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data={
"text":[
"I am very happy today",
"I feel amazing and joyful",
"This is the best day",
"I love my life",
"I am feeling very sad",
"I feel depressed and lonely",
"This is a terrible day",
"I am crying",
"I am very angry",
"This makes me furious",
"I hate this situation",
"I am so mad right now",
"I am scared",
"This is frightening",
"I feel nervous and afraid",
"The dark place scares me",
"I feel fantastic today",
"What a wonderful day",
"I feel miserable",
"I am extremely upset",
"I am furious about this",
"This makes me rage",
"I feel terrified",
"I am frightened right now"
],
"emotion":[
"happy","happy","happy","happy",
"sad","sad","sad","sad",
"angry","angry","angry","angry",
"fear","fear","fear","fear","happy","happy",
"sad","sad",
"angry","angry",
"fear","fear"
]
}



df=pd.DataFrame(data)

print(df.head())
print(df.info())
print(df["emotion"].value_counts())

choice = input("Show dataset graph? (yes/no): ")

if choice == "yes":
    df["emotion"].value_counts().sort_index().plot(kind="bar")
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()

X=df["text"]
y=df["emotion"]



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)

vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

model=MultinomialNB()
model.fit(X_train_vec,y_train)

pred=model.predict(X_test_vec)


print("Accuracy:",accuracy_score(y_test,pred))


while True:
    text=input("Enter text: ")
    text_vec=vectorizer.transform([text])
    emotion=model.predict(text_vec)
    print("Predicted Emotion:",emotion[0])
```
<img width="1205" height="637" alt="image" src="https://github.com/user-attachments/assets/73875d40-9dc3-48ef-8eed-479384d79c7c" />
<img width="795" height="683" alt="image" src="https://github.com/user-attachments/assets/85854b54-6850-451a-aaed-8b7890b41192" />


