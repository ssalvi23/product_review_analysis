from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from reviewscrap import scrape_all_reviews


def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def analyze_reviews(product_url):
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

        results = []
        for i in range(average_scores.shape[0]):
            l = labels[ranking[i]]
            s = np.round(float(average_scores[ranking[i]]), 4)
            results.append({'label': l, 'score': s})

        return results
    else:
        return []


# Example usage:
product_url = "https://www.amazon.in/Lifelong-Liquidizing-Stainless-Manufacturers-LLMG900/dp/B0C1KHZ3GL/ref=sr_1_6?crid=B20R76GT7XT8&dib=eyJ2IjoiMSJ9.ydujPfLmO6pmghZ7dbEijt5jYkKtfPuoVoHmJjtZCv8hnUrj8FKxJKhkFDsMoiCnW8R0by5jzeODY9Lvd65jTsk-8z8ucFN81Mu4j7dDIQSNzCQ8_QPVnsdOKb_53E0iFF01Y04gVsbFvmZ7Gla-liwQLEySj8UoPUJqtCYzUC9lFBBP2UsmBK3LaMC01vaMDwC2VeggHJS7g2OxGSVOYxyzvDfL5QafAj5PncvEr2k.wEsSakNvXPh74DRQ5ofXSqPFtYTiLJ-wkPOm6eN--Ec&dib_tag=se&keywords=mixer+grinder+under+2k&qid=1712376126&sprefix=mixer+grinder+under+2+k%2Caps%2C244&sr=8-6"
results = analyze_reviews(product_url)
print(results)
