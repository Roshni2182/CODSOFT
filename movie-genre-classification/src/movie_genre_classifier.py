import os
import random
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "movie_genre.csv")

data = pd.read_csv(DATA_PATH)

# ================= RANDOMNESS =================
# shuffle dataset every run
data = data.sample(frac=1).reset_index(drop=True)

X = data["plot"]
y = data["genre"]

# random test size each run
test_size = random.choice([0.2, 0.25, 0.3])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size
)

# TF-IDF with randomness
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=random.choice([3000, 4000, 5000])
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy (test_size={test_size}):", round(accuracy, 3))

# ================= RANDOM TEST INPUT =================
test_plots = [
    "A hero saves the city from dangerous criminals",
    "A couple falls in love during college life",
    "A ghost haunts a group of teenagers",
    "A detective investigates a mysterious murder",
    "Friends go on a hilarious road trip"
]

random_plot = random.choice(test_plots)
prediction = model.predict(tfidf.transform([random_plot]))[0]

print("Random Test Plot:", random_plot)
print("Predicted Genre:", prediction)

# ================= SAVE MODEL =================
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, "model", "genre_model.pkl"))
joblib.dump(tfidf, os.path.join(BASE_DIR, "model", "tfidf.pkl"))
