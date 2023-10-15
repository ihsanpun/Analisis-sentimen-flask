from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
from jcopml.utils import load_model
from flask import Flask, render_template, request
import string
import re
from nltk.tokenize import word_tokenize

app = Flask(__name__)
model = load_model("analisis2.pkl")

# Define a function to clean and preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and URLs
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(cleaned_tokens)

def apply_sastrawi_stemming(text):
    stemmer = StemmerFactory().create_stemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

@app.route("/csv", methods=["GET", "POST"])
def csv():
    if request.method == "GET":
        return render_template("upload.html")
    elif request.method == "POST":
        csv_file = request.files.get("file")
        X_test = pd.read_csv(csv_file)
        X_test["clean"] = X_test["tweet"].apply(preprocess_text)
        X_test["stopwords_removed"] = X_test["clean"].apply(remove_stopwords)
        X_test["stem"] = X_test["stopwords_removed"].apply(apply_sastrawi_stemming)
        predictions = model.predict(X_test["stem"])
        X_test["pred"] = predictions

        return X_test.to_html()

@app.route("/tweet", methods=["GET", "POST"])
def tweet():
    if request.method == "GET":
        return render_template("form.html")
    elif request.method == "POST":
        text = request.form.get("name")  # Remove the list wrapping
        clean = preprocess_text(text)
        clean = remove_stopwords(clean)
        clean = apply_sastrawi_stemming(clean)
        pred = model.predict([clean])  # Wrap the cleaned text in a list
        return str(pred[0])  # Return the prediction as a string

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
