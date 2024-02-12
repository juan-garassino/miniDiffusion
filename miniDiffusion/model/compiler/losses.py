from tensorflow.math import reduce_mean


def loss_fn(real, generated):
    loss = reduce_mean((real - generated) ** 2)
    return loss
