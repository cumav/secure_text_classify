"""
If using this in colab, use:
    !pip install tensorflow_text
    !pip install pandas
    !pip install keras
to install needed dependencies.
"""

import urllib.request
import os
import zipfile
import random

import tensorflow_hub as hub
import tensorflow as tf
tf.sysconfig.get_link_flags()
import pandas as pd
import tensorflow_text
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
dump_dir = os.path.join(dir_path, "dumpdata")
data_dir = os.path.join(dir_path, "data")
download_filename = os.path.join(data_dir, "trainingandtestdata.zip")
tf_cache_dir = os.path.join(data_dir, "pretrained_model")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TFHUB_CACHE_DIR'] = tf_cache_dir

# Creating path if directories do not exist
for path in [dump_dir, data_dir]:
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created directory {} ...".format(path))

if not os.path.exists(download_filename):
    data_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    print("Downloading dataset ...")
    urllib.request.urlretrieve(data_url, download_filename)

    print("Extracting dataset ...")
    with zipfile.ZipFile(download_filename, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

# Start creating embeddings
data = pd.read_csv(os.path.join(data_dir, "training.1600000.processed.noemoticon.csv"), header=None, encoding='latin-1')
data = data.drop(labels=[1, 2, 3, 4], axis=1)
data.columns = ["category", "text"]

print("Loading model. If this is the first time running this script, this may take some time ...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2")

# Compute embeddings.
dict_data = data.to_dict('records')
counter = 0
for dd in dict_data:
    dd["id"] = counter
    counter += 1
print("Finished converting data to an array of dicts - Example text: {}".format(dict_data[0]["text"]))
print("shuffeling")
random.shuffle(dict_data)

steps = 2000
for num in range(0, len(dict_data), steps):

    text = [x["text"] for x in dict_data[num: num + steps]]
    categories = [x["category"] for x in dict_data[num: num + steps]]
    sentiment_id = [x["id"] for x in dict_data[num: num + steps]]

    en_result = embed(text)["outputs"]
    print("writing to file")

    list_of_data = []
    out_data = en_result.numpy()
    for i in range(len(out_data)):
        # TODO: Check if its usefull to pass the id. Else remove the list comp. for the id
        temp = {'text': text[i], 'embedding': out_data[i], 'category': categories[i]}
        list_of_data.append(temp)
    pickle.dump(list_of_data, open(os.path.join(dump_dir, "{}.p".format(num)), "wb"))
