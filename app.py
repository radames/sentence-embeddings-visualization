from umap_reducer import UMAPReducer
from embeddings_encoder import EmbeddingsEncoder
from flask import Flask, request, render_template, jsonify, make_response, session
from flask_session import Session
from flask_cors import CORS, cross_origin
import os
from dotenv import load_dotenv
import feedparser
import json
from dateutil import parser
import re
import numpy as np
import gzip
import hashlib

load_dotenv()


app = Flask(__name__, static_url_path='/static')
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") 
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
Session(app)
CORS(app)

reducer = UMAPReducer()
encoder = EmbeddingsEncoder()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run-umap', methods=['POST'])
@cross_origin(supports_credentials=True)
def run_umap():
    input_data = request.get_json()
    sentences = input_data['data']['sentences']
    umap_options = input_data['data']['umap_options']
    cluster_options = input_data['data']['cluster_options']
    # create unique hash for input, avoid recalculating embeddings
    sentences_input_hash = hashlib.sha256(
        ''.join(sentences).encode("utf-8")).hexdigest()

    print("input options:", sentences_input_hash,
          umap_options, cluster_options, "\n\n")
    try:
        if not session.get(sentences_input_hash):
            print("New input, calculating embeddings" "\n\n")
            embeddings = encoder.encode(sentences)
            session[sentences_input_hash] = embeddings.tolist()
        else:
            print("Input already calculated, using cached embeddings", "\n\n")
            embeddings = session[sentences_input_hash]

        # UMAP embeddings
        reducer.setParams(umap_options, cluster_options)
        umap_embeddings = reducer.embed(embeddings)
        # HDBScan cluster analysis
        clusters = reducer.clusterAnalysis(umap_embeddings)
        content = gzip.compress(json.dumps(
            {
                "embeddings": umap_embeddings.tolist(),
                "clusters": clusters.labels_.tolist()
            }
        ).encode('utf8'), 5)
        response = make_response(content)
        response.headers['Content-length'] = len(content)
        response.headers['Content-Encoding'] = 'gzip'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=int(os.environ.get('PORT', 7860)))
