import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import Fore, Style
from PIL import Image
import os

from miniDiffusion.managers.manager import Manager
from miniDiffusion.tools.params import one_minus_sqrt_alpha_bar, sqrt_alpha_bar, timesteps

def set_noise_generator_seed(seed, verbose=False):
    """
    Sets the seed for the random noise generator to ensure reproducibility.

    This function initializes the random number generator (RNG) seed for NumPy.
    It ensures that the same noise is generated for a given time step,
    enabling consistent noise addition during the denoising diffusion process.

    Parameters:
    - seed: int, the seed value for initializing the RNG.
    - verbose: bool, whether to display descriptive messages.
    """
    np.random.seed(seed)

    # Print descriptive message if verbose is True
    if verbose:
        print("\n🔽 " + Fore.GREEN + "Noise generator seed has been set for reproducibility" + Style.RESET_ALL)

def generate_noising_timestamps(seed, num, verbose=False):
    """
    Generates a batch of sample timestamps for the denoising diffusion process.

    This function generates a batch of sample timestamps using a provided seed value
    to initialize the random number generator (RNG). These timestamps are used during
    the denoising diffusion process for modeling.

    Parameters:
    - seed: int, seed value for initializing the RNG.
    - num: int, number of timestamps to generate.
    - verbose: bool, whether to display descriptive messages.

    Returns:
    - timestamps: TensorFlow tensor, batch of generated timestamps.
    """

    set_noise_generator_seed(seed)

    # Generate timestamps
    timestamps = tf.random.uniform(shape=[num], minval=0, maxval=timesteps, dtype=tf.int32)

    # Print descriptive message if verbose is True
    if verbose:
        print("\n🔽 " + Fore.GREEN + "Batch of timestamps has been successfully generated" + Style.RESET_ALL)

        print("\n🔽 " + Fore.BLUE + f"Generated Timestamps: {timestamps}" + Style.RESET_ALL)

    return timestamps

def forward_noise(key, x_0, timestep, verbose=False):
    """
    Adds noise to the input image based on the given timestamp.

    Parameters:
    - key: int, seed value for initializing the RNG.
    - x_0: numpy array, input image.
    - timestep: int, timestamp used to determine the noise level.
    - verbose: bool, if True, plots noise and noisy image.

    Returns:
    - noisy_image: numpy array, noisy version of the input image.
    - noise: numpy array, generated noise.
    """

    set_noise_generator_seed(key)

    noise = np.random.normal(size=x_0.shape)

    reshaped_sqrt_alpha_bar_t = np.reshape(np.take(sqrt_alpha_bar, timestep), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, timestep), (-1, 1, 1, 1))

    noisy_image = reshaped_sqrt_alpha_bar_t * x_0 + reshaped_one_minus_sqrt_alpha_bar_t * noise

    if verbose:
        # Select one image from the batch
        image_index = 0  # You can change this index to visualize a different image from the batch
        single_noise = noise[image_index]
        single_noisy_image = noisy_image[image_index]

        # Plot the selected image, its corresponding noise, and the noisy version
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(x_0[image_index].squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(single_noise.squeeze(), cmap='gray')
        axes[1].set_title(f'Noise {timestep}')
        axes[1].axis('off')

        axes[2].imshow(single_noisy_image.squeeze(), cmap='gray')
        axes[2].set_title('Noisy Image')
        axes[2].axis('off')

        # Define the directory and filename for saving the image
        out_dir = Manager.working_directory('training_data')
        Manager.make_directory(out_dir)
        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plot_filename = f"{out_dir}/noisy_image_plot[{now}].png"

        # Save the plot with timestamp in the filename
        plt.savefig(plot_filename)

        # Close the plot to avoid cache problems
        plt.close()

        # Print descriptive message
        print("\n🔽 " + Fore.GREEN + "Noisy image plot has been saved at: " + plot_filename + Style.RESET_ALL)

    return noisy_image, noise

def generate_noising_picture_snapshots(dataset, colab=0, steps_range=100):
    """
    Generates and saves snapshots of noised images at different timesteps.

    Parameters:
    - dataset: TensorFlow dataset containing the images.
    - colab: int, indicates whether the code is running on Colab.
    - steps_range: int, range of timesteps to visualize.

    Returns:
    None
    """
    # Retrieve a sample from the dataset
    sample_mnist = next(iter(dataset))[0]

    # Define the output directory
    out_dir = Manager.working_directory("snapshots", colab=colab)

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over the specified range of steps
    for i in range(steps_range):
        # Generate random seeds for the RNG
        rng, tsrng = np.random.randint(0, 100000, size=(2,))

        # Generate noised image and noise
        noised_im, noise = forward_noise(
            rng,
            np.expand_dims(sample_mnist, 0),
            np.array([i]),
        )

        # Create a new figure for each step
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image, added noise, and noised image
        axes[0].imshow(sample_mnist, cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(np.squeeze(noise), cmap="gray")
        axes[1].set_title("Added Noise")

        axes[2].imshow(np.squeeze(noised_im), cmap="gray")
        axes[2].set_title(f"Noised Image @ {i + 1}")

        # Save plot to snapshot directory
        snapshot_path = os.path.join(out_dir, f"snapshot_{i:04d}.png")
        plt.savefig(snapshot_path)
        plt.close()

    print("\n🔽 " + Fore.GREEN + f"Noised image snapshots saved in: {out_dir}" + Style.RESET_ALL)

def gif_from_directoty(input_path, output_path, interval=200):
    # Get a list of image file names in the input directory
    image_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    # Sort image files based on their names
    image_files.sort()

    # Load images from the input directory
    imgs = []

    for file_name in image_files:
        file_path = os.path.join(input_path, file_name)
        img = Image.open(file_path)
        img.load()  # Ensure image data is fully loaded
        imgs.append(img)

    # Save images as GIF
    if not os.path.isdir(output_path):  # Check if output_path is a directory
        imgs[0].save(
            fp=output_path,
            format="GIF",
            append_images=imgs[1:],
            save_all=True,
            duration=interval,
            loop=0
        )

        print("\n🔽 " + Fore.BLUE +
              f"Generated GIF of {len(imgs)} frames from {input_path} to {output_path}" +
              Style.RESET_ALL)
    else:
        print("\n🔺 " + Fore.RED +
              f"Error: Output path {output_path} is a directory. Provide a valid file path." +
              Style.RESET_ALL)

def save_gif(img_list, path="", interval=200):
    # Transform images from [-1,1] to [0, 255]
    imgs = []
    for im in img_list:
        im = np.array(im)
        im = (im + 1) * 127.5
        im = np.clip(im, 0, 255).astype(np.int32)
        im = Image.fromarray(im)
        imgs.append(im)

    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(
        fp=path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=interval,
        loop=0,
    )

    print("\n🔽 " + Fore.BLUE +
          "Genearted gif of {} frames from {}".format(interval, path) +
          Style.RESET_ALL)

if __name__ == "__main__":

    from miniDiffusion.managers.manager import Manager

    manager = Manager()  # Initialize the project manager

    dataset = manager.get_datasets(dataset='mnist', samples=1000, batch_size=1)  # Project manager loads the data

    generate_noising_picture_snapshots(dataset, colab=0, steps_range=1000)

    input_path='/Users/juan-garassino/Results/miniDiffusion/snapshots'

    output_path='/Users/juan-garassino/Results/animation.gif'

    gif_from_directoty(input_path, output_path, interval=200)
