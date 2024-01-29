import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from miniDiffusion.utils.params import one_minus_sqrt_alpha_bar, sqrt_alpha_bar, timesteps

def set_key(key):
    """
    Sets the random number generator (RNG) seed for NumPy.

    Parameters:
    - key: int, seed value for initializing the RNG.
    """
    np.random.seed(key)

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
        axes[1].set_title('Noise')
        axes[1].axis('off')

        axes[2].imshow(single_noisy_image.squeeze(), cmap='gray')
        axes[2].set_title('Noisy Image')
        axes[2].axis('off')

        # Generate a timestamp for the plot filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save the plot with timestamp in the filename
        plot_filename = f"noisy_image_plot_{timestamp}.png"
        plt.savefig(plot_filename)

        # Close the plot to avoid cache problems
        plt.close()

    return noisy_image, noise

def generate_timestamp(key, num):
    """
    Generates a batch of sample timestamps.

    Parameters:
    - key: int, seed value for initializing the RNG.
    - num: int, number of timestamps to generate.

    Returns:
    - timestamps: TensorFlow tensor, batch of generated timestamps.
    """
    set_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=timesteps, dtype=tf.int32)
