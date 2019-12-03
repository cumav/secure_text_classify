import os
import pickle
import glob

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
import numpy as np

from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(dir_path, "logs", now_time)
models_dir = os.path.join(dir_path, "models")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

tensorboard = TensorBoard(log_dir=logdir)

def load_pickle_data(folder):
    all_data = []

    for item in glob.glob("{}\*".format(folder)):
        temp_data = pickle.load(open(item, "rb"))
        all_data += temp_data
    return all_data

data = load_pickle_data(os.path.join(dir_path, "dumpdata"))

message_embeddings = np.array([item["embedding"] for item in data])
zero_or_one = lambda x: 1 if x == 4 else 0
labels = np.array([zero_or_one(item["category"]) for item in data])


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=512))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(message_embeddings, labels,
          epochs=20,
          batch_size=64,
          verbose=2,
          validation_split=0.2,
          callbacks=[tensorboard])

model.save(os.path.join(models_dir, "{}.h5".format(now_time)))