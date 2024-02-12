import tensorflow as tf

# Normalization helper
def preprocess(x, y):
    return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))


def reshape():
    pass


def crop():
    pass
