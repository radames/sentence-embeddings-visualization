---
title: Sentence Embeddings Visualization
emoji: 📈
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Hugging Face Spaces + Observable
### Sentence Embeddings Visualization

Recently I've been exploring [Hugging face Spaces](https://huggingface.co/spaces) and [sentence-transformers](https://huggingface.co/sentence-transformers) to build an application to generate text embeddings and clustering visualization.

Currently, the quickest way to build interactive ML apps with Python (backend/frontend), afaik, is to use [Streamlit](https://streamlit.io/) or [Gradio](https://www.gradio.app/). To embed an Observable notebook on Streamlit, you can use this custom component [streamlit-observable](https://github.com/asg017/streamlit-observable) 

This [Observable notebook](https://observablehq.com/@radames/hugging-face-spaces-observable-sentence-embeddings) is the frontend application for this [Hugging Face Spaces](https://huggingface.co/spaces/radames/sentence-embeddings-visualization) app.

This notebook explores another way to integrate Observable inside Hugging Face Spaces. Currently,  [HF Spaces supports](https://huggingface.co/docs/hub/spaces#streamlit-and-gradio) Streamlit and Gradio or a simple static web page. 

The concept here is to use this entire notebook as the frontend and data visualization application for the [ML Flask/Python](https://huggingface.co/spaces/radames/sentence-embeddings-visualization/blob/main/app.py#L37-L75) backend.

* The index route renders a [simple HTML template](https://huggingface.co/spaces/radames/sentence-embeddings-visualization/blob/main/templates/index.html) containing [Observable Runtime API code](https://observablehq.com/@observablehq/downloading-and-embedding-notebooks).
* A single function, triggered by a POST request to \`run-umap\`, returns a low dimensional representation of the original sentence transformers embeddings using UMAP and cluster analysis with HDBSCAN.
* All the visualization and interactive magic happen on the Javascript code inside the Observable Notebook.
