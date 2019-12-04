# Secure text analysis

This project aims to create a privacy focused text analysis tool. For this, the following steps are planned:

- Use a Tensorflow hub model to create embeddings from the text to be classified. (Maybe train it with own data). The embeddings then will be send to a remote server for classification. Maybe port the molde to java/C#/go?
- Generate a model for:
    * Sentiment classification
    * Spam classification
    * Topic classification
    
- For optimal models https://autokeras.com/ should be researched.
- Run the classifyer model in the cloud.

![Process](media/process.svg)

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

`tensorboard --logdir=./logs``
