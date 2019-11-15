import tensorflow as tf
import tensorflow_hub as hub
import os

from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TFHUB_CACHE_DIR'] = 'C:\\path\\to\\pretraiend_model\\sms_spam_tf_hub\\pretrained_model'

data = pd.read_csv("spam.csv", encoding='latin-1')
data = data.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data.columns = ["category", "text"]
labels = data.category

for num, item in enumerate(labels):
    if item == "ham":
        labels[num] = 0
    else:
        labels[num] = 1


# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(
    "https://tfhub.dev/google/universal-sentence-encoder-large/3")

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embed(list(data.text)))


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=512))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_size = int(len(data)*0.8)

model.fit(message_embeddings[:train_size], labels[:train_size], epochs=10, batch_size=32)

# eval model accuracy
loss, acc = model.evaluate(message_embeddings[train_size:], data.category[train_size:], verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))