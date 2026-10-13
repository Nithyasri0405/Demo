import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("heart_disease_uci.csv")

# convert target to binary (0 = no disease, 1 = disease)
data['target'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# convert all features into text format
def convert_to_text(row):
    text = ""
    for col in data.columns:
        if col != 'num' and col != 'target':
            text += col + "_" + str(row[col]) + " "
    return text

data['text'] = data.apply(convert_to_text, axis=1)

X = data['text']
y = data['target']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train model
model = MultinomialNB()
model.fit(X_train, y_train)

# check accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# take user input
print("\nEnter patient details")

age = input("Age: ")
sex = input("Sex (1=Male, 0=Female): ")
cp = input("Chest Pain (0-3): ")
chol = input("Cholesterol: ")
thalach = input("Max Heart Rate: ")

# convert input to same format
user_data = "age_" + age + " sex_" + sex + " cp_" + cp + " chol_" + chol + " thalach_" + thalach

# predict
user_vec = vectorizer.transform([user_data])
result = model.predict(user_vec)

# output
if result[0] == 1:
    print("Disease Present")
else:
    print("No Disease")