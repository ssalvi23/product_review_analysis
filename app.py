from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from reviewscrap import scrape_all_reviews

app = Flask(__name__)

def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(product_url):
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    reviewarray = scrape_all_reviews(product_url)

    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

    average_scores = np.zeros(len(labels))
    total_reviews = len(reviewarray)

    if total_reviews > 0:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        for review in reviewarray:
            text = preprocess(review)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            average_scores += scores

        average_scores /= total_reviews

        ranking = np.argsort(average_scores)
        ranking = ranking[::-1]
        results = [{'label': labels[ranking[i]], 'score': np.round(float(average_scores[ranking[i]]), 4)} for i in range(average_scores.shape[0])]
        return results
    else:
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        product_url = request.form['product_url']
        results = analyze_sentiment(product_url)
        return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
