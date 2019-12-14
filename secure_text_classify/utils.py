def load_tf_hub_model(model_name="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/2"):
    import tensorflow_hub as hub
    import tensorflow as tf
    tf.sysconfig.get_link_flags()
    import tensorflow_text

    return hub.load(model_name)
