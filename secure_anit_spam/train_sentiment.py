import os
import pickle
import glob

from keras import Sequential
from keras.layers import Dense
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_pickle_data(folder):
    all_data = []

    for item in glob.glob("{}\*".format(folder)):
        temp_data = pickle.load(open(item, "rb"))
        all_data += temp_data
    return all_data

data = load_pickle_data(os.path.join(dir_path, "dumpdata"))

message_embeddings = np.array([item["embedding"] for item in data])
zero_or_one = lambda x : 1 if x == 4 else 0
labels = np.array([zero_or_one(item["category"]) for item in data])


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=512))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_size = int(len(data)*0.8)

model.fit(message_embeddings[:train_size], labels[:train_size], epochs=10, batch_size=32, verbose=2)

# eval model accuracy
loss, acc = model.evaluate(message_embeddings[train_size:], labels[train_size:], verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))