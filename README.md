# Secure text analysis

This project aims to create a privacy focused text analysis tool, without exposing classification models to the end user.
For this we use so called embeddings. 
Embeddings are a vector representation of words, sentences or phrases. You can still extract information 
out of them, though the exact information of the text is lost. This is where the privacy factor comes in. We can use
the embeddings as a numerical 'summary' (in this case 520 floats) of the text, which is not backtrackable. 
The embeddings only represent the most basic information of a given text (like topics), which we can then safely send to a 
backend server.

The process would look as follows:

1) Convert sentences to embedding vectors using a TensorflowHub model
2) Send embeddings to backend
3) Classify embeddings (e.g. sentiment, spam, topic)
4) Send back classifications to frontend


![Process](media/process.svg)

For this, the following steps are planned:

[x] Use a [TensorFlow hub model](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2) to create embeddings from the text to be classified (Maybe fine tune it by transfer learning).
 The embeddings then will be send to a remote server for classification. 
 
[] The embeddings model needs to be converted to be usable with Java/Javascript/C#, so that the embeddings get
generated by the frontend.

[] Generate a model for:

    * Sentiment classification [x]
    * Spam classification []
    * Topic classification [x]
    

### Dev setup:

As of today, pip has to be on the latest version for everything to work corectly:

``pip install -U pip``

Keep in mind that becuase the module ``tensorflow_text``, this project does 
not work on Windows.

Setup venv:

```pip -m venv env```

Install package in development mode:

``pip install -e .``

Using TensorBoard:

* Search for the logdir. Here its "./logs".

``tensorboard --logdir=./logs``

### Example use:

Load training data and train a models on multiple parameters:


```python
from secure_text_classify import train_sentiment
import os

# Initialize Trainer
x = train_sentiment.Train(os.path.dirname(os.path.realpath(__file__)))
# Downloads and prepares data
x.prepare_data()

# Create multiple models with different dropouts and nodes
dropout = [0.2, 0.3, 0.4, 0.6]
nodes = [12, 32, 64, 128]
x.train_on_ranges(dropout, nodes)
```

After training the models select the best one, then you can 
use it to create a simple API:
```python 
from secure_text_classify import serve
import os

# "20191212-145717_nodes-128_dropout-0.3.h5" is the model with the best validation accuracy
serve.serve(os.path.dirname(os.path.realpath(__file__)), "20191212-145717_nodes-128_dropout-0.3.h5")
```

Now you can POST sentences to ``localhost:5000``:

```JSON
{
	"text": "This is fun!"
}
```
It will return a float value where 1 is a very positive sentiment and 0 is a negative sentiment. 

## Dockerization of the application

The dockerfiles are based on [tiangolo's](https://github.com/tiangolo) docker images
containing nginx, uwsgi and flask. The Dockerfiles had to be modified, to support tensorflow.

To generate the base image ``cd`` into the folder ``docker-tensorflow``:

``cd docker-tensorflow``

Then start building the image by running:

``docker build .``

After the image has been build, select the image id and tag it:

``docker tag <image-id> tensorflow/nginx:latest``

To find the image-id type ``docker images`` it should be the latest one.

Now cd back into this projects directory and build the image:

``cd ../``

``docker build .``

When finished building, select the image-id (not "tensorflow/nginx:latest") and run a container:

```docker container run --publish 80:80 <image-id>```

or to run it in the background:

``docker container run --publish 80:80 detached <image-id>``
