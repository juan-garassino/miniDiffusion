import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import Fore, Style
from miniDiffusion.managers.manager import Manager


from miniDiffusion.utils.params import one_minus_sqrt_alpha_bar, sqrt_alpha_bar, timesteps

def set_key(key, verbose=False):
    """
    Sets the random number generator (RNG) seed for NumPy.

    Parameters:
    - key: int, seed value for initializing the RNG.
    - verbose: bool, whether to display descriptive messages.
    """
    np.random.seed(key)

    # Print descriptive message if verbose is True
    if verbose:
        print("\n🔽 " + Fore.GREEN + "RNG seed has been successfully set for NumPy" + Style.RESET_ALL)

def forward_noise(key, x_0, t, verbose=False):
    """
    Adds noise to the input image based on the given timestamp.

    Parameters:
    - key: int, seed value for initializing the RNG.
    - x_0: numpy array, input image.
    - t: int, timestamp used to determine the noise level.
    - verbose: bool, if True, plots noise and noisy image.

    Returns:
    - noisy_image: numpy array, noisy version of the input image.
    - noise: numpy array, generated noise.
    """
    set_key(key)

    noise = np.random.normal(size=x_0.shape)

    reshaped_sqrt_alpha_bar_t = np.reshape(np.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))

    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))

    noisy_image = (reshaped_sqrt_alpha_bar_t * x_0 + reshaped_one_minus_sqrt_alpha_bar_t * noise)

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
        axes[1].set_title(f'Noise {t[image_index]}')
        axes[1].axis('off')

        axes[2].imshow(single_noisy_image.squeeze(), cmap='gray')
        axes[2].set_title('Noisy Image')
        axes[2].axis('off')

        # Define the directory and filename for saving the image.
        out_dir = Manager.working_directory('training_data')

        Manager.make_directory(out_dir)

        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        plot_filename = f"{out_dir}/noisy_image_plot[{now}].png"

        # Generate a timestamp for the plot filename
        # Save the plot with timestamp in the filename
        plt.savefig(plot_filename)

        # Close the plot to avoid cache problems
        plt.close()

    return noisy_image, noise

def generate_timestamp(key, num, verbose=False):
    """
    Generates a batch of sample timestamps.

    Parameters:
    - key: int, seed value for initializing the RNG.
    - num: int, number of timestamps to generate.
    - verbose: bool, whether to display descriptive messages.

    Returns:
    - timestamps: TensorFlow tensor, batch of generated timestamps.
    """

    set_key(key)

    # Generate timestamps
    timestamps = tf.random.uniform(shape=[num], minval=0, maxval=timesteps, dtype=tf.int32)

    # Print descriptive message if verbose is True
    if verbose:
        print("\n🔽 " + Fore.GREEN + "Batch of timestamps has been successfully generated" + Style.RESET_ALL)

    return timestamps
