import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from colorama import Fore, Style
from miniDiffusion.managers.denoiser import denoising_diffusion_implicit_models
from miniDiffusion.managers.manager import Manager
from miniDiffusion.model.model import Unet


def create_blank_image_for_inpainting(
    original_image,
    blank_size,
    verbose=False,
    save=False,
    save_path="modified_image.png",
):
    """
    Creates a blank section within an image for inpainting.

    Parameters:
    - original_image: numpy array, the original image with shape (batch_size, height, width, channels).
    - blank_size: tuple, (width, height) defining the size of the blank area.
    - verbose: bool, whether to display the modified image.
    - save: bool, whether to save the modified image.
    - save_path: str, the file path to save the modified image.

    Returns:
    - numpy array: original image with a blank section for inpainting.
    """
    # Create a copy of the original image to avoid modifying the original.
    modified_image = np.copy(original_image)

    # Extract the dimensions of the original image.
    batch_size, image_height, image_width, _ = original_image.shape

    # Extract the size of the blank area.
    blank_width, blank_height = blank_size

    # Iterate over each image in the batch.
    for i in range(batch_size):
        # Calculate the coordinates for the center of the blank area.
        center_x = image_width // 2
        center_y = image_height // 2

        # Calculate the coordinates for the top-left corner of the blank area.
        x_start = max(0, center_x - blank_width // 2)
        y_start = max(0, center_y - blank_height // 2)

        # Calculate the coordinates for the bottom-right corner of the blank area.
        x_end = min(image_width, x_start + blank_width)
        y_end = min(image_height, y_start + blank_height)

        # Set the specified blank area to zeros for the current image.
        modified_image[i, y_start:y_end, x_start:x_end, :] = 0

    # Display the modified image if verbose is True.
    if verbose:
        plt.imshow(modified_image[0])
        plt.show()

    # Save the modified image if save is True.
    if save:
        # Save the modified image using Matplotlib
        plt.imshow(modified_image[0])
        plt.savefig(save_path)

    return modified_image


def inpaint_with_ddim(
    original_image,
    unet_model,
    blank_size=(6, 6),
    total_timesteps=100,
    inference_steps=10,
    verbose=False,
    save_interval=None,
):
    """
    Utilizes the denoising diffusion implicit model for inpainting.

    Parameters:
    - unet_model: tf.keras.Model, the U-Net model used for predicting noise.
    - total_timesteps: int, total number of timesteps in the denoising diffusion process.
    - starting_noise: numpy array, starting point of noise.
    - inference_steps: int, number of inference steps within each denoising segment.
    - verbose: bool, whether to display intermediate images.
    - save_interval: int, specifies the interval for saving images.

    Returns:
    None
    """

    image_with_blank = create_blank_image_for_inpainting(original_image, blank_size)

    print(image_with_blank.shape)

    plt.imshow(image_with_blank.reshape((32, 32, 1)))

    plt.show()

    # Perform denoising diffusion implicit model for inpainting.
    denoising_diffusion_implicit_models(
        unet_model=unet_model,
        total_timesteps=total_timesteps,
        starting_noise=image_with_blank,
        inference_steps=inference_steps,
        verbose=verbose,
        save_interval=save_interval,
    )


def set_model_trainable(model, trainable=False):
    """
    Set all layers of the model to trainable or non-trainable.

    Parameters:
    - model: tf.keras.Model, the model whose layers' trainability will be set.
    - trainable: bool, if True, set all layers to trainable, otherwise set them to non-trainable.
    """
    for layer in model.layers:
        layer.trainable = trainable


if __name__ == "__main__":

    try:

        manager = (
            Manager()
        )  # unet, optimizer, args.DATA)  # Initialize the project manager

        # Get a single sample from the dataset
        original_image_iterator = iter(
            manager.get_datasets(dataset="mnist", samples=1, batch_size=1)
        )

        original_image = next(original_image_iterator)

        # Define your model and optimizer
        unet_model = Unet(channels=1)  # Instantiate your U-Net model

        optimizer = tf.keras.optimizers.Adam()  # Choose your optimizer

        directory = Manager.working_directory("checkpoints", colab=0)

        # Define your checkpoint objects
        checkpoint = tf.train.Checkpoint(model=unet_model, optimizer=optimizer)

        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=directory, max_to_keep=5
        )

        # Check if there is a checkpoint to restore, otherwise train from scratch
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print("Checkpoint restored from", checkpoint_manager.latest_checkpoint)
        else:
            print("No checkpoint found. Training from scratch.")

        print(original_image.shape)

        set_model_trainable(unet_model, trainable=False)

        # Perform inpainting using the denoising diffusion implicit model
        inpaint_with_ddim(
            original_image,
            unet_model=unet_model,
            blank_size=(5, 20),
            total_timesteps=100,
            inference_steps=100,
            verbose=True,
            save_interval=10,
        )

    except:

        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

    # Example usage:
    # inpaint_with_ddim(unet_model, total_timesteps=100, starting_noise=None, inference_steps=10, verbose=False, save_interval=None)
