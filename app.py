from umap_reducer import UMAPReducer
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS
import os
from dotenv import load_dotenv
from transformers import pipeline
import feedparser
import json
from dateutil import parser
import re
import numpy as np
import gzip

load_dotenv()

# Load Setiment Classifier
# sentiment_analysis = pipeline(
#     "sentiment-analysis", model="siebert/sentiment-roberta-large-english")
app = Flask(__name__, static_url_path='/static')
reducer = UMAPReducer()

CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run-umap')  # //methods=['POST'])
def run_umap():
    data = np.random.rand(512, 4)

    # UMAP embeddings
    embeddings = reducer.embed(data)

    content = gzip.compress(json.dumps(embeddings.tolist()).encode('utf8'), 5)
    response = make_response(content)
    response.headers['Content-length'] = len(content)
    response.headers['Content-Encoding'] = 'gzip'
    return response


# @app.route('/news')
# def get_news():
#     feed_url = request.args.get('feed_url')
#     # check if string is a valid

#     # file name for cache
#     file_name = "".join(re.split(r"https://|\.|/", feed_url))

#     feed_entries = get_feed(feed_url)
#     # filter only titles for sentiment analysis
#     try:
#         with open(f'{file_name}_cache.json') as file:
#             cache = json.load(file)
#     except:
#         cache = {}

#     # if new homepage is newer than cache, update cache and return
#     print("new date", feed_entries['last_update'])
#     print("old date", cache['last_update']
#           if 'last_update' in cache else "None")
#     if not cache or parser.parse(feed_entries['last_update']) > parser.parse(cache['last_update']):
#         print("Updating cache with new preditions")
#         titles = [entry['title'] for entry in feed_entries['entries']]
#         # run sentiment analysis on titles
#         predictions = [sentiment_analysis(sentence) for sentence in titles]
#         # parse Negative and Positive, normalize to -1 to 1
#         predictions = [-prediction[0]['score'] if prediction[0]['label'] ==
#                        'NEGATIVE' else prediction[0]['score'] for prediction in predictions]
#         # merge rss data with predictions
#         entries_predicitons = [{**entry, 'sentiment': prediction}
#                                for entry, prediction in zip(feed_entries['entries'], predictions)]
#         output = {'entries': entries_predicitons,
#                   'last_update': feed_entries['last_update']}
#         # update last precitions cache
#         with open(f'{file_name}_cache.json', 'w') as file:
#             json.dump(output, file)
#         # send back json
#         return jsonify(output)
#     else:
#         print("Returning cached predictions")
#         return jsonify(cache)


# @ app.route('/predict', methods=['POST'])
# def predict():
#     # get data from POST
#     if request.method == 'POST':
#         # get current news
#         # get post body data
#         data = request.get_json()
#         if data.get('sentences') is None:
#             return jsonify({'error': 'No text provided'})
#         # get post expeceted to be under {'sentences': ['text': '...']}
#         sentences = data.get('sentences')
#         # prencit sentiments
#         predictions = [sentiment_analysis(sentence) for sentence in sentences]
#         # parse Negative and Positive, normalize to -1 to 1
#         predictions = [-prediction[0]['score'] if prediction[0]['label'] ==
#                        'NEGATIVE' else prediction[0]['score'] for prediction in predictions]
#         output = [dict(sentence=sentence, sentiment=prediction)
#                   for sentence, prediction in zip(sentences, predictions)]
#         # send back json
#         return jsonify(output)


# def get_feed(feed_url):
#     feed = feedparser.parse(feed_url)
#     return {'entries': feed['entries'], 'last_update': feed["feed"]['updated']}
if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=int(os.environ.get('PORT', 7860)))
