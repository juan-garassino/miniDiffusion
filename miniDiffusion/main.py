from turtle import width
from colorama import Fore, Style
import os
import numpy as np
import time

from miniDiffusion.models.model import Unet
from miniDiffusion.models.losses import loss_fn
from miniDiffusion.models.optimizer import optimizer

from miniDiffusion.managers.denoiser import (
    denoising_diffusion_implicit_models,
    denoising_diffusion_probabilistic_models,
    denoise_process,
)
from miniDiffusion.managers.manager import Manager

from miniDiffusion.utils.utils import generate_timestamp, forward_noise

from tensorflow import GradientTape, get_logger, Variable
from tensorflow.keras.utils import Progbar
from tensorflow.train import Checkpoint, CheckpointManager


get_logger().setLevel("ERROR")  # Suppressing tf.hub warnings

unet = Unet(channels=1)  # Create our unet model

output_directory = Manager.working_directory("checkpoints")  # Directory

Manager.make_directory(output_directory)  # Makes the directory

manager = Manager(unet, optimizer,
                  os.environ.get("DATA"))  # Initialize the project manager

dataset = manager.get_datasets()  # Project manager loads the data

checkpoint = Checkpoint(step=Variable(1), optimizer=optimizer,
                        model=unet)  # Creates the checkpoint

checkpoint_manager = CheckpointManager(
    checkpoint, output_directory,
    max_to_keep=3)  # Creates a checkpoint manager

manager.train_and_checkpoint(checkpoint,
    checkpoint_manager)  # Checkpoint manager loads the last checkpoint

test_images = np.ones([1, 32, 32,
                       1])  # initialize the model in the memory of our GPU

test_timestamps = generate_timestamp(0, 1)

k = unet(test_images, test_timestamps)

rng = 0

def train_step(batch):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))

    print("\n🔽 " + Fore.GREEN + f"RNG: {rng}, TSRNG: {tsrng}" + Style.RESET_ALL)

    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    print("\n🔽 " + Fore.GREEN + f"Timestep Values Shape: {timestep_values.shape} [{timestep_values[0]} ... {timestep_values[-1]}]" + Style.RESET_ALL)

    noised_image, noise = forward_noise(rng, batch, timestep_values, verbose=True)

    with GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)

        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, unet.trainable_variables)

    optimizer.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value


for epoch in range(1, int(os.environ.get("EPOCHS")) + 1):

    start = time.time()
    # this is cool utility in Tensorflow that will create a nice looking progress bar
    bar = Progbar(len(dataset) - 1, width=50)  # keras progress bar!!

    losses = []

    print(
        "\n⏩ "
        + Fore.MAGENTA
        + f"Training diffusion model for epoch {epoch} of {int(os.environ.get('EPOCHS'))}\n" + "\n", end="",
    )

    for i, batch in enumerate(iter(dataset)):
        # run the training loop
        loss = train_step(batch)
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    print(Style.RESET_ALL)

    save_path = checkpoint_manager.save()

    print("✅ " + Fore.CYAN + "Saved checkpoint for step {}: {}".format(
        int(checkpoint.step), save_path) + Style.RESET_ALL)


    avg = np.mean(losses)

    print(
        "\n📶 " + Fore.CYAN +
        f"Average loss for epoch {epoch}/{int(os.environ.get('EPOCHS'))}: {avg}"
        + Style.RESET_ALL)

    print(
        "\n✅ "
        + Fore.CYAN
        + "Time for epoch {} is {} sec".format(epoch, time.time() - start)
        + Style.RESET_ALL
    )

denoise_process(unet)  # Invalid denoising environment specified.

# denoising_diffusion_implicit_models(unet, timesteps=100, starting_noise=None, verbose=False, save_interval=None)

# denoising_diffusion_implicit_models(unet)
