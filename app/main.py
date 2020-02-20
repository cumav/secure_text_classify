import json
import os
import pickle

from flask import Flask, request
from flask_cors import CORS
from keras.models import load_model
import nltk
from scipy import spatial
from itertools import islice

nltk.download('punkt')

from secure_text_classify.utils import load_tf_hub_model

app = Flask(__name__)
CORS(app)

cache_dir = os.path.dirname(os.path.realpath(__file__))
sentiment_model_name = os.path.join("sentiment", "20191212-145717_nodes-128_dropout-0.3.h5")
# Load embeddings model from TFHUB

tf_cache_dir = os.path.join(cache_dir, "pretrained_model")
os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
embed = load_tf_hub_model()

# Load trained keras model for sentiment analysis
model_path = os.path.join(cache_dir, "models", sentiment_model_name)
sentiment_model = load_model(model_path)

# load emoji embeddings
with open(os.path.join(cache_dir, "models", "emoji_embeddings.pickle"), 'rb') as handle:
    emoji_embeddings = pickle.load(handle)


def n_smallest_cosine_similarity(base_embedding, embeddings_dict, n_ranking=5):
    """
    Will calculate the best n closest embeddings from embeddings_dict to base_embedding. Will use cosine similarty.
    The closer the embedding of embeddings dict to base_embedding, the better.
    """
    _loss_dict = {}
    for item in embeddings_dict:
        loss = spatial.distance.cosine(base_embedding, embeddings_dict[item])
        _loss_dict[item] = loss

    ordered_loss_dict = {k: v for k, v in sorted(_loss_dict.items(), key=lambda item: item[1])}
    return list(islice(ordered_loss_dict.items(), n_ranking))


@app.route('/', methods=['GET', 'POST'])
def predict():
    global embed
    global sentiment_model
    data = request.data
    request_data = json.loads(data)
    sentence_list = nltk.tokenize.sent_tokenize(request_data["text"])
    embedding = embed(sentence_list)["outputs"]
    predictions = sentiment_model.predict(embedding.numpy())
    embedding_data = {}
    for num, item in enumerate(predictions):
        topic_emojis = n_smallest_cosine_similarity(embedding[num], emoji_embeddings)
        embedding_data[num] = {"text": sentence_list[num], "value": float(item[0]), "emojis": topic_emojis}

    return embedding_data


if __name__ == "__main__":
    """
    Starts a flask server to serve the models
    :cache_dir: Path where the models are stored
    :sentiment_model_name: Name of the model
    """

    app.run(host='0.0.0.0', threaded=False, port=80)
