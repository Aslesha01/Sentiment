import numpy as np
import pandas as pd
import pickle

np.random.seed(1)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import re
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")  # downloading database
nltk.download("averaged_perceptron_tagger")  # parts of speech tagger
from nltk.corpus import wordnet  # database on which lemmatizer works
from nltk.stem import WordNetLemmatizer  # lemmatizer function
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv("AirlineTweets.csv")

positive_reviews = [
    "The flight was fantastic!",
    "I had an amazing experience with the airline.",
    "The service provided was outstanding.",
    "The food was decent",
    "The environment was pretty decent"
]

# Load the existing DataFrame from the CSV file
filename = "AirlineTweets.csv"  # Replace with the actual filename
df = pd.read_csv(filename)

# Create a new DataFrame with the positive reviews
new_reviews_df = pd.DataFrame(
    {"airline_sentiment": "positive", "text": positive_reviews}
)

# Append the new reviews DataFrame to the existing DataFrame
df = pd.concat([df, new_reviews_df], ignore_index=True)

df = df[["text", "airline_sentiment"]]

df["airline_sentiment"] = df["airline_sentiment"].apply(
    lambda x: 1 if x == "positive" else 0
)


def clean_text(text_series):
    # Convert to lowercase
    text_series = text_series.str.lower()

    # Remove punctuation
    text_series = text_series.str.replace(r"[^\w\s]", "")

    # Remove numbers
    text_series = text_series.str.replace(r"\d+", "")

    # Tokenize the text
    tokens = text_series.apply(word_tokenize)

    # Remove stopwords
    additional_stopwords = {"the", "was", "flight"}
    stop_words = set(stopwords.words("english"))
    stop_words = stop_words.union(additional_stopwords)
    tokens = tokens.apply(lambda x: [token for token in x if token not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

    # print(text_series[14642],tokens.loc[14642])

    cleaned_text = tokens.apply(lambda x: " ".join(x))

    return cleaned_text


cleaned_text = clean_text(df.text)

df_train, df_test, y_train, y_test = train_test_split(
    cleaned_text, df.airline_sentiment, test_size=0.2, random_state=42
)

vec = TfidfVectorizer(max_features=2000)

X_train = vec.fit_transform(df_train)
X_test = vec.transform(df_test)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy_test = accuracy_score(y_test, predictions)
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print(accuracy_train, accuracy_test)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
