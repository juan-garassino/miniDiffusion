from miniDiffusion.models.model import Unet
from miniDiffusion.utils.utils import generate_timestamp, forward_noise
from miniDiffusion.models.losses import loss_fn
from miniDiffusion.utils.data import get_datasets
from miniDiffusion.utils.params import data

from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import time
from colorama import Fore, Style

dataset = get_datasets()



# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# create our unet model
unet = Unet(channels=1)

# create our checkopint manager
ckpt = tf.train.Checkpoint(unet=unet)
ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=2)

# load from a previous checkpoint if it exists, else initialize the model from scratch
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])

    print("\n🔽 " + Fore.BLUE +
          "Restored from {}".format(ckpt_manager.latest_checkpoint) +
          Style.RESET_ALL)

else:

    print("\n⏹ " + Fore.GREEN + "Initializing from scratch." + Style.RESET_ALL)

# initialize the model in the memory of our GPU
test_images = np.ones([1, 32, 32, 1])
test_timestamps = generate_timestamp(0, 1)
k = unet(test_images, test_timestamps)

# create our optimizer, we will use adam with a Learning rate of 1e-4
opt = Adam(learning_rate=1e-4)


rng = 0


def train_step(batch):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    noised_image, noise = forward_noise(rng, batch, timestep_values)
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)

        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value


epochs = 10
for epoch in range(1, epochs + 1):

    start = time.time()
    # this is cool utility in Tensorflow that will create a nice looking progress bar
    bar = tf.keras.utils.Progbar(len(dataset) - 1) # keras progress bar!!
    losses = []

    print("\n⏩ " + Fore.MAGENTA + f"Training diffusion model for epoch {epoch}\n" + "\n", end="")

    for i, batch in enumerate(iter(dataset)):
        # run the training loop
        loss = train_step(batch)
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    print(Style.RESET_ALL + '\n')

    avg = np.mean(losses)

    print("\n📶 " + Fore.CYAN + f"Average loss for epoch {epoch}/{epochs}: {avg}" +
          Style.RESET_ALL)

    print("\n✅ " + Fore.CYAN +
          "Time for epoch {} is {} sec".format(epoch + 1,
                                               time.time() - start) +
          Style.RESET_ALL)
