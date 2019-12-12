import operator
import os
import pickle
import glob

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.models import load_model
import numpy as np

from datetime import datetime


class Train:

    def __init__(self, dir_path=os.path.dirname(os.path.realpath(__file__))):
        self.dir_path = dir_path
        self.models_dir = os.path.join(self.dir_path, "models")
        self.dumpdata_dir = os.path.join(self.dir_path, "dumpdata")

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        data = self.load_pickle_data(self.dumpdata_dir)

        zero_or_one = lambda x: 1 if x == 4 else 0

        self.message_embeddings = np.array([item["embedding"] for item in data])
        self.labels = np.array([zero_or_one(item["category"]) for item in data])

    @staticmethod
    def load_pickle_data(folder):
        """
        Loads the pickled data, containing embeddings and classes.
        :param folder: Name of the folder containing the pickled embeddings.
        :return: Array containing all embeddings in this format:
            [
                {"embeddings": array([0, .., n=521]), "id": 1, "category": 0},
                ...
                {"embeddings": array([0, .., n=521]), "id": n, "category": 4}
            ]
        """
        all_data = []

        for item in glob.glob(os.path.join(folder, "*")):
            temp_data = pickle.load(open(item, "rb"))
            all_data += temp_data
        return all_data

    def make_model(self, num_nodes, dropout):
        """
        Create the model architecture. Only droput and number of nodes can be changed.
        Since we want a binary classification we only need one output node.
        More layers most certainly will result in overfitting.
        :param num_nodes: Number of nodes in the hidden layer
        :param dropout: Dropout "strength"
        """

        self.model = Sequential()
        self.model.add(Dense(num_nodes, activation='relu', input_dim=512))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train_and_saved(self, num_nodes=32, dropout=0.4, epochs=15):
        """
        Saves the the model and gives it the name of the current time. In TensorBoard
        the best model can then be selected.
        :return: model name, models validation accuracy
        """
        self.make_model(num_nodes, dropout)

        model_id = "{}_nodes-{}_dropout-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"), num_nodes, dropout)
        logdir = os.path.join(self.dir_path, "logs", model_id)

        tensorboard = TensorBoard(log_dir=logdir)

        fit_stats = self.model.fit(self.message_embeddings, self.labels,
                                   epochs=epochs,
                                   batch_size=64,
                                   verbose=2,
                                   validation_split=0.2,
                                   callbacks=[tensorboard])
        self.model.save(os.path.join(self.models_dir, "{}.h5".format(model_id)))

        return model_id, fit_stats.history["val_accuracy"][-1]

    def validate(self, best_model):
        """
        Check if model can be loaded and print the accuracy of the model using all data
        :param best_model: model to check functionality for.
        """

        data = self.load_pickle_data(self.dumpdata_dir)
        embeddings = np.array([x["embedding"] for x in data])
        labels = [x["category"] for x in data]
        model = load_model(best_model)
        predictions = model.predict(embeddings)

        right_pred = 0
        for num, pred in enumerate(predictions):
            if pred[0] < 0.5:
                if labels[num] == 0:
                    right_pred += 1
            else:
                if labels[num] == 4:
                    right_pred += 1

        print(right_pred / len(labels))

    def train_on_ranges(self, dropouts, nodes):
        """
        Will iter over stated dropouts and nodes and will return the best model
        :param dropouts: Array of dropouts
        :param nodes: Array where each item represents the number of nodes the hidden layer has
        :return: best model name
        """

        model_stats = {}

        for node_count in nodes:
            for dropout in dropouts:
                print("\nTraining model with dropout of: {} and with a "
                      "number of nodes of: {}\n".format(dropout, node_count))
                model_name, model_val_accuracy = self.train_and_saved(num_nodes=node_count, dropout=dropout, epochs=15)
                model_stats[model_name] = model_val_accuracy

        sorted_stats = sorted(model_stats.items(), key=operator.itemgetter(1))

        for val, item in enumerate(sorted_stats):
            print("{}. place is model '{}' with an accuracy of: {}\n".format(val, item[0], item[1]))

        # TODO: continue retrain the exiting model
        best_model = os.path.join(self.models_dir, "{}.h5".format(sorted_stats[-1][0]))
        model = load_model(best_model)
        # TODO: Fit here

    def prepare_data(self):
        """
        Generates training data for the sentiment classifier
        """
        from secure_text_classify.prepare_sentiment_data import prepare_sentiment_data
        prepare_sentiment_data(dir_path=self.dir_path)


def main():
    dropout = [0.2, 0.3, 0.4, 0.6]
    nodes = [12, 32, 64, 128]
    x = Train()
    x.prepare_data()
    x.train_on_ranges(dropout, nodes)


if __name__ == "__main__":
    main()
