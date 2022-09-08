from miniDiffusion.models.model import Unet
from miniDiffusion.models.losses import loss_fn
from miniDiffusion.models.optimizer import optimizer

from miniDiffusion.managers.denoiser import denoising_diffusion_implicit_models, denoising_diffusion_probabilistic_models
from miniDiffusion.managers.manager import Manager

from miniDiffusion.utils.utils import generate_timestamp, forward_noise
from miniDiffusion.data.data import get_datasets

from colorama import Fore, Style

import numpy as np
from tensorflow import GradientTape, get_logger
from tensorflow.keras.utils import Progbar
import time



dataset = get_datasets()

# Suppressing tf.hub warnings
get_logger().setLevel("ERROR")

# create our unet model
unet = Unet(channels=1)

manager = Manager(unet)

manager.load_model()

# initialize the model in the memory of our GPU
test_images = np.ones([1, 32, 32, 1])
test_timestamps = generate_timestamp(0, 1)
k = unet(test_images, test_timestamps)

# create our optimizer, we will use adam with a Learning rate of 1e-4

rng = 0

def train_step(batch):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    noised_image, noise = forward_noise(rng, batch, timestep_values)
    with GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)

        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, unet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value

epochs = 10

for epoch in range(1, epochs + 1):

    start = time.time()
    # this is cool utility in Tensorflow that will create a nice looking progress bar
    bar = Progbar(len(dataset) - 1)  # keras progress bar!!
    losses = []

    print(
        "\n⏩ " + Fore.MAGENTA + f"Training diffusion model for epoch {epoch} of {epochs}\n" + "\n",
        end="",
    )

    for i, batch in enumerate(iter(dataset)):
        # run the training loop
        loss = train_step(batch)
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    print(Style.RESET_ALL)

    avg = np.mean(losses)

    print(
        "\n📶 "
        + Fore.CYAN
        + f"Average loss for epoch {epoch}/{epochs}: {avg}"
        + Style.RESET_ALL
    )

    print(
        "\n✅ "
        + Fore.CYAN
        + "Time for epoch {} is {} sec".format(epoch, time.time() - start)
        + Style.RESET_ALL
    )

denoising_diffusion_probabilistic_models(unet)

denoising_diffusion_implicit_models(unet)
