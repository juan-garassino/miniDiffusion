from turtle import width
from colorama import Fore, Style
import os
import numpy as np
import time
import sys
import argparse

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

def train_step(batch, verbose=False):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))

    if verbose:
        print("\n🔽 " + Fore.GREEN + f"RNG: {rng}, TSRNG: {tsrng}" + Style.RESET_ALL, end="")

    timestep_values = generate_timestamp(tsrng, batch.shape[0], verbose=False)

    if verbose:
        print("\n🔽 " + Fore.GREEN + f"Timestep Values Shape: {timestep_values.shape} [{timestep_values[0]} ... {timestep_values[-1]}]" + Style.RESET_ALL, end="")

    noised_image, noise = forward_noise(rng, batch, timestep_values, verbose=False)

    with GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)
        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, unet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value

def parse_arguments():

    parser = argparse.ArgumentParser(description="Process arguments for running the Python file.")

    # Define command-line arguments with defaults
    parser.add_argument("--COLAB", type=int, default=0, help="Description of COLAB")
    parser.add_argument("--EPOCHS", type=int, default=1, help="Description of EPOCHS")
    parser.add_argument("--DENOISING", type=str, default='DDPM', help="Description of DENOISING")
    parser.add_argument("--DATA", type=str, default='mnist', help="Description of DATA")
    parser.add_argument("--N_SAMPLES", type=int, default=500, help="Description of N_SAMPLES")
    parser.add_argument("--BATCH_SIZE", type=int, default=64, help="Description of BATCH_SIZE")
    parser.add_argument("--BUFFER", type=int, default=128, help="Description of BUFFER")
    parser.add_argument("--TIMESTEPS", type=int, default=400, help="Description of TIMESTEPS")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_arguments()

    # Printing parsed arguments with color and emojis in one line
    print(
        f"\n🆗{' '.join([f'{Fore.MAGENTA}{k}:{v}' for k, v in vars(args).items()])}\n{Style.RESET_ALL}"
    )

    get_logger().setLevel("ERROR")  # Suppressing tf.hub warnings

    unet = Unet(channels=1)  # Create our unet model

    output_directory = Manager.working_directory("checkpoints", colab=args.COLAB)  # Directory

    Manager.make_directory(output_directory)  # Makes the directory

    manager = Manager(unet, optimizer, args.DATA)  # Initialize the project manager

    dataset = manager.get_datasets(dataset='mnist', samples=50000, batch_size=64)  # Project manager loads the data

    checkpoint = Checkpoint(step=Variable(1), optimizer=optimizer, model=unet)  # Creates the checkpoint

    checkpoint_manager = CheckpointManager(checkpoint, output_directory, max_to_keep=3)  # Creates a checkpoint manager

    manager.train_and_checkpoint(checkpoint, checkpoint_manager)  # Checkpoint manager loads the last checkpoint

    test_images = np.ones([1, 32, 32, 1])  # initialize the model in the memory of our GPU

    test_timestamps = generate_timestamp(0, 1, verbose=False)

    k = unet(test_images, test_timestamps)

    rng = 0

    for epoch in range(1, int(args.EPOCHS) + 1):

        start = time.time()
        # this is cool utility in Tensorflow that will create a nice looking progress bar
        bar = Progbar(len(dataset) - 1, width=50)  # keras progress bar!!

        losses = []

        print(
            "\n⏩ "
            + Fore.MAGENTA
            + f"Training diffusion model for epoch {epoch} of {int(args.EPOCHS)}\n" + "\n", end="",
        )

        for i, batch in enumerate(iter(dataset)):
            # run the training loop
            loss = train_step(batch, verbose=False)
            losses.append(loss)
            bar.update(i, values=[("loss", loss)])

        print(Style.RESET_ALL)

        save_path = checkpoint_manager.save()

        print("✅ " + Fore.CYAN + "Saved checkpoint for step {}: {}".format(
            int(checkpoint.step), save_path) + Style.RESET_ALL)


        avg = np.mean(losses)

        print(
            "\n📶 " + Fore.CYAN +
            f"Average loss for epoch {epoch}/{int(args.EPOCHS)}: {avg}"
            + Style.RESET_ALL)

        print(
            "\n✅ "
            + Fore.CYAN
            + "Time for epoch {} is {} sec".format(epoch, time.time() - start)
            + Style.RESET_ALL
        )

    denoise_process(unet, denoising_method=args.DENOISING, timesteps=100, starting_noise=None, verbose=True, save_interval=None)
