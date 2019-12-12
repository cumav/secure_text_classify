import json
import os

from flask import Flask, request

from secure_text_classify.utils import load_tf_hub_model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    data = request.data
    request_data = json.loads(data)
    embedding = embed([request_data["text"]])["outputs"]
    embedding_array = embedding.numpy().tolist()
    embedding_data = {"result": embedding_array}
    return embedding_data


def serve(cache_dir):
    global embed
    tf_cache_dir = os.path.join(cache_dir, "pretrained_model")
    os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    embed = load_tf_hub_model()
    app.run()
