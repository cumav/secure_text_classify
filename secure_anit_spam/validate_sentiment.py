from keras.models import load_model
import glob
import pickle
import numpy as np

def load_pickle_data(folder):
    all_data = []

    for item in glob.glob("{}\*".format(folder)):
        temp_data = pickle.load(open(item, "rb"))
        all_data += temp_data
    return all_data

data = load_pickle_data("./dumpdata")
embeddings = np.array([x["embedding"] for x in data])
labels = [x["category"] for x in data]
model = load_model("./models/20191203-211700.h5")
predictions = model.predict(embeddings)

right_pred = 0
for num, pred in enumerate(predictions):
    if pred[0] < 0.5:
        if labels[num] == 0:
            right_pred += 1
    else:
        if labels[num] == 4:
            right_pred += 1

print(right_pred/len(labels))
