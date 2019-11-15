# Secure auto spam classifier

This project aims to create a privacy focused spam classifier. For this, the following steps are planned:

- Use a Tensorflow hub model to create embeddings from the text to be classified. (Maybe train it with own data). The embeddings then will be send to a remote server for classification. Maybe port the molde to java/C#/go?
- Generate a model using https://autokeras.com/  for classifying the embeddings.
- Run the classifyer model in the cloud.

### Dev setup:
Setup venv:

```pip -m venv env```

Install package in development mode:

``pip install -e .``

