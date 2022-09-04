def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss
