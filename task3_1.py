import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np

data = {
    "symptom_text": [
        "mild headache and sneezing",
        "high fever and chest pain",
        "slight throat irritation",
        "difficulty breathing and fatigue",
        "runny nose and cough",
        "severe abdominal pain and vomiting",
        "occasional sore throat",
        "intense headache and blurred vision",
        "mild skin rash on arms",
        "persistent cough and body ache",
        "nausea and stomach cramps",
        "light muscle soreness",
        "fever and chills",
        "severe back pain and dizziness",
        "headache after long travel"
    ],
    "severity": [
        "mild", "severe", "mild", "severe", "mild",
        "severe", "mild", "severe", "mild", "severe",
        "severe", "mild", "severe", "severe", "mild"
    ]
}

df = pd.DataFrame(data)
df["severity"] = (df["severity"].str.strip().str.lower() == "mild").astype(int)

train, temp = train_test_split(df, test_size=0.4)
valid, test = train_test_split(temp, test_size=0.5)

X_train, y_train = train["symptom_text"], train["severity"]
X_valid, y_valid = valid["symptom_text"], valid["severity"]
X_test, y_test = test["symptom_text"], test["severity"]

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)
X_test_vec = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

user_input = input("Enter your symptoms: ")
user_vec = vectorizer.transform([user_input])
prediction = nb_model.predict(user_vec)
confidence = np.max(nb_model.predict_proba(user_vec))

severity_label = "Mild" if prediction == 1 else "Severe"
print(f"Severity: {severity_label} (Confidence: {confidence:.2f})")
