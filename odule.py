import os

import numpy as np
import pandas as pd



data = pd.read_csv("./sentiment.csv", header=None, encoding='latin-1')
data = data.drop(labels=[1, 2, 3, 4], axis=1)
data.columns = ["category", "text"]
labels = data.category
y = [0 for x in range(len(labels))]

for num, item in enumerate(labels):
    if int(item) == 0:
        y[num] = 0
    else:
        y[num] = 1

print("Pos/Neg share = {}".format(sum(y)/len(y)))

# Compute embeddings.
last_num = 0
counter = 0
print("im here")

dict_data = data.to_dict()
print(dict_data["text"][1])
