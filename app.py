from umap_reducer import UMAPReducer
from embeddings_encoder import EmbeddingsEncoder
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS
import os
from dotenv import load_dotenv
import feedparser
import json
from dateutil import parser
import re
import numpy as np
import gzip

load_dotenv()


app = Flask(__name__, static_url_path='/static')
reducer = UMAPReducer()
encoder = EmbeddingsEncoder()
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run-umap', methods=['POST'])
def run_umap():
    input_data = request.get_json()
    sentences = input_data['data']['sentences']
    umap_options = input_data['data']['umap_options']
    cluster_options = input_data['data']['cluster_options']

    print("input options:", umap_options, cluster_options)
    try:
        embeddings = encoder.encode(sentences)
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
        return jsonify({"error": str(e)}), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=int(os.environ.get('PORT', 7860)))
