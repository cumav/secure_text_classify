import json
import os

from flask import Flask, request
import numpy as np
from keras.models import load_model

from secure_text_classify.utils import load_tf_hub_model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    data = request.data
    request_data = json.loads(data)
    embedding = embed([request_data["text"]])["outputs"]
    predictions = sentiment_model.predict(embedding.numpy())
    embedding_data = {}
    for num, item in enumerate(predictions):
        embedding_data[num] = {"text": request_data["text"], "value": float(item[0])}

    return embedding_data


def serve(cache_dir, sentiment_model_name):
    """
    Starts a flask server to serve the models
    :cache_dir: Path where the models are stored
    :sentiment_model_name: Name of the model
    """
    # Load embeddings model from TFHUB
    global embed
    tf_cache_dir = os.path.join(cache_dir, "pretrained_model")
    os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    embed = load_tf_hub_model()

    # Load trained keras model for sentiment analysis
    global sentiment_model
    model_path = os.path.join(cache_dir, "models", sentiment_model_name)
    sentiment_model = load_model(model_path)

    app.run(threaded=False)