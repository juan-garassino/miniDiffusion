"""
Denoising Diffusion Probabilistic Models for Image Generation

This Python file presents implementations of two advanced generative models: Denoising
Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM).
These models represent cutting-edge techniques in the field of image generation, offering
a novel approach to create high-quality images from noise through a process resembling
thermodynamic diffusion.

Key Components:

1. denoising_diffusion_probabilistic_models: This function demonstrates the process of
   generating images using DDPM. It iteratively applies a denoising process to transform
   noise into coherent images, leveraging a U-Net model for noise prediction. The function
   also includes mechanisms for saving and displaying intermediate results, offering
   insights into the model's progression.

2. denoising_diffusion_implicit_models: Similar to its counterpart, this function employs
   DDIM, a variant of DDPM, for image generation. DDIM offers a faster sampling process,
   which makes it more suitable for applications where speed is crucial. The function
   iteratively generates images, providing visualization at each step.

Highlights:

- Both models are examples of the latest advancements in generative deep learning,
  offering an alternative to traditional GAN-based approaches.

- The implementations here are designed to be modular and easily integrated into broader
  machine learning pipelines, particularly those using TensorFlow.

- These models open new possibilities for high-quality image synthesis, applicable in
  various domains such as art generation, data augmentation, and more.

Usage:

To utilize these models, TensorFlow and other dependencies should be properly set up.
Users can adapt the code to their specific needs, experimenting with different
configurations and observing the impact on the generated images. The code is structured
to provide clear insights into each step of the diffusion process, making it educational
for those new to the concept of diffusion models.

This file serves as both a practical tool for generating images using diffusion models
and a learning resource for understanding these complex yet fascinating models.
"""

# Following this comment, the implementation of the denoising_diffusion_probabilistic_models, denoising_diffusion_implicit_models, and any supporting functions or classes would be placed.

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from colorama import Fore, Style
from PIL import Image

from miniDiffusion.tools.utils import save_gif
from miniDiffusion.managers.manager import Manager
from miniDiffusion.tools.params import alpha, alpha_bar, beta


def ddpm_denoise(input_data, predicted_noise, timestep):
    """
    Denoises the input data `input_data` using the DDPM (Diffusion Denoising Probabilistic Model) method at timestep `timestep`.

    Parameters:
    - input_data: numpy array, input data at timestep `timestep`, possibly corrupted by noise.
    - predicted_noise: numpy array, noise predicted by the diffusion model.
    - timestep: int, timestep at which the denoising process is applied.

    Returns:
    - denoised_data: numpy array, denoised version of input data `input_data` at timestep `timestep`.

    The function computes the denoised version of the input data `input_data` at timestep `timestep` using the DDPM method.
    It extracts `alpha_t` and `alpha_t_bar` from predefined sequences `alpha` and `alpha_bar` respectively,
    according to the timestep `timestep`, which are part of the noise schedule in the diffusion process.
    The coefficient for scaling the predicted noise, `eps_coef`, is calculated based on `alpha_t` and `alpha_t_bar`.
    The mean of the reverse diffusion process at timestep `timestep` is computed to reconstruct the data from the noisy version.
    The variance at timestep `timestep` is obtained from the predefined sequence `beta`.
    Noise is sampled from a normal distribution to add stochasticity to the reverse process.
    The denoised data is returned with added noise according to the variance at timestep `timestep`.
    """

    # Extract `alpha_t` and `alpha_t_bar` from pre-defined sequences `alpha` and `alpha_bar`
    # using the timestep `timestep`. These values are part of the noise schedule in the diffusion process.
    alpha_t = np.take(alpha, timestep)
    alpha_t_bar = np.take(alpha_bar, timestep)

    # Calculate the coefficient for scaling the predicted noise.
    eps_coef = (1 - alpha_t) / np.sqrt(1 - alpha_t_bar)

    # Compute the mean of the reverse diffusion process at timestep `timestep`.
    # This is part of the denoising step, reconstructing the data from the noisy version.
    mean = (1 / np.sqrt(alpha_t)) * (input_data - eps_coef * predicted_noise)

    # Variance at timestep `timestep` is obtained from the pre-defined sequence `beta`.
    variance_timestep = np.take(beta, timestep)

    # Sample noise from a normal distribution to add stochasticity to the reverse process.
    noise_sample = np.random.normal(size=input_data.shape)

    # Return the denoised data with added noise according to the variance at timestep `timestep`.
    return mean + np.sqrt(variance_timestep) * noise_sample


def ddim_denoise(input_data, predicted_noise, timestep, noise_std):
    """
    Performs denoising and diffusion-based image modeling (DDIM) at timestep `timestep`.

    Parameters:
    - input_data: numpy array, input data at timestep `timestep`, possibly corrupted by noise.
    - predicted_noise: numpy array, noise predicted by the diffusion model.
    - timestep: int, timestep at which the denoising process is applied.
    - noise_std: float, standard deviation of the noise distribution at timestep `timestep`.

    Returns:
    - denoised_prediction: numpy array, denoised prediction for timestep `timestep`.

    The function computes the denoised prediction for the input data `input_data` at timestep `timestep` using the DDIM method.
    It extracts `alpha_t_bar` and `alpha_t_minus_one` from predefined sequences `alpha_bar` and `alpha` respectively,
    according to the timestep `timestep`, which are part of the noise schedule in the diffusion process.
    The prediction is computed by denoising and adjusting for the alpha coefficients.
    Additional adjustments are made by adding a scaled version of the predicted noise and additional noise
    scaled by `noise_std` for stochasticity.
    The final prediction for timestep `timestep` is returned.
    """

    # Extract `alpha_t_bar` and `alpha_t_minus_one` using timestep `timestep` from pre-defined sequences.
    alpha_t_bar = np.take(alpha_bar, timestep)
    alpha_t_minus_one = np.take(alpha, timestep - 1)

    # Compute the prediction by denoising and adjusting for the alpha coefficients.
    prediction = (input_data - ((1 - alpha_t_bar) ** 0.5) * predicted_noise) / (
        alpha_t_bar**0.5
    )
    prediction = (alpha_t_minus_one**0.5) * prediction

    # Adjust the prediction by adding a scaled version of the predicted noise.
    prediction = (
        prediction
        + ((1 - alpha_t_minus_one - (noise_std**2)) ** 0.5) * predicted_noise
    )

    # Add additional noise scaled by `noise_std` for stochasticity.
    additional_noise = np.random.normal(size=input_data.shape)
    prediction = prediction + (noise_std * additional_noise)

    # Return the final prediction for the timestep `timestep`.
    return prediction


def denoising_diffusion_probabilistic_models(
    unet_model,
    total_timesteps=100,
    starting_noise=None,
    verbose=False,
    save_interval=None,
    colab=0,
):
    """
    Performs the Denoising Diffusion Probabilistic Model process.

    Parameters:
    - unet_model: tf.keras.Model, the U-Net model used for predicting noise.
    - total_timesteps: int, number of timesteps in the diffusion process.
    - starting_noise: numpy array, starting point of noise.
    - verbose: bool, whether to display intermediate images.
    - save_interval: int, specifies the interval for saving images.

    Returns:
    None
    """

    # Starting the Denoising Diffusion Probabilistic Model process.
    print(
        "\n⏹ "
        + Fore.GREEN
        + "Denoising Diffusion Probabilistic Model started"
        + Style.RESET_ALL
    )

    # Initialize a random noise image if starting_noise is not provided.
    if starting_noise is None:
        noise_image = tf.random.normal((1, 32, 32, 1))
    else:
        noise_image = starting_noise

    # Store the initial noise image.
    image_list = [np.squeeze(np.squeeze(noise_image, 0), -1)]

    # Define the initial picture name as None.
    picture_name = None

    # Iteratively apply the denoising diffusion process.
    for i in range(total_timesteps - 1):
        # Calculate the current timestep.
        current_timestep = np.expand_dims(
            np.array(total_timesteps - i - 1, np.int32), 0
        )

        # Predict the noise using the U-Net model.
        predicted_noise = unet_model(noise_image, current_timestep)

        # Apply the denoising diffusion process.
        noise_image = ddpm_denoise(noise_image, predicted_noise, current_timestep)

        # Append the generated image for later use.
        image_list.append(np.squeeze(np.squeeze(noise_image, 0), -1))

        # Update the picture_name within the loop.
        output_directory = Manager.working_directory("snapshots", colab=colab)
        Manager.make_directory(output_directory)
        current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        picture_name = f"{output_directory}/image[{current_time}][{i}].png"

        # Save the current generated image at specified intervals and at the end.
        if save_interval is not None and (
            i % save_interval == 0 or i == total_timesteps - 2
        ):
            # Display the generated image if verbose is True.
            if verbose:
                plt.imshow(
                    np.array(
                        np.clip((noise_image[0] + 1) * 127.5, 0, 255)[:, :, 0], np.uint8
                    ),
                    cmap="gray",
                )
                plt.show()

            temp = (
                np.array(
                    np.clip(
                        tf.reshape(noise_image[0], shape=(32, 32)), a_min=0, a_max=255
                    )
                    + 1
                )
                * 127.5
            )  # np.uint8

            # Save the image.
            Image.fromarray(temp, mode="L").save(picture_name)

            # Print the status message.
            print(
                "\n🔽 "
                + Fore.BLUE
                + f"Generated picture {picture_name.split('/')[-1]} @ {output_directory}"
                + Style.RESET_ALL
            )

    # Generate and save the final animation as a GIF.
    save_gif(image_list + ([image_list[-1]] * 100), picture_name, interval=20)

    # Display and print the final generated image if verbose is True and picture_name is assigned.
    if verbose and picture_name is not None:
        plt.imshow(
            np.array(np.clip((noise_image[0] + 1) * 127.5, 0, 255)[:, :, 0], np.uint8)
        )
        plt.show()
        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Generated gif {picture_name.split('/')[-1]} at {output_directory}"
            + "\n"
            + Style.RESET_ALL
        )


def denoising_diffusion_implicit_models(
    unet_model,
    total_timesteps=100,
    starting_noise=None,
    inference_steps=10,
    verbose=False,
    save_interval=None,
    colab=0,
):
    """
    Performs the Denoising Diffusion Implicit Model process.

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

    # Starting the Denoising Diffusion Implicit Model process.
    print(
        "\n⏹ "
        + Fore.GREEN
        + "Denoising Diffusion Implicit Model started"
        + Style.RESET_ALL
    )

    # Define the range of inference steps based on total timesteps and inference steps.
    inference_range = range(0, total_timesteps, total_timesteps // inference_steps)

    # Initialize a random noise image if starting_noise is not provided.
    if starting_noise is None:
        noise_image = tf.random.normal((1, 32, 32, 1))
    else:
        noise_image = starting_noise

    # Store the initial noise image.
    image_list = [np.squeeze(np.squeeze(noise_image, 0), -1)]

    # Define the name of the generated image file initially as None.
    generated_image_file = None

    # Iteratively apply the denoising diffusion process.
    for index, step_index in enumerate(reversed(range(inference_steps))):
        # Calculate the current timestep.
        current_step = np.expand_dims(inference_range[step_index], 0)

        # Predict the noise using the U-Net model.
        predicted_noise = unet_model(noise_image, current_step)

        # Apply denoising using the predicted noise.
        noise_image = ddim_denoise(noise_image, predicted_noise, current_step, 0)

        # Append the denoised image to the list.
        image_list.append(np.squeeze(np.squeeze(noise_image, 0), -1))

        # Update the file path for the generated image.
        output_directory = Manager.working_directory("snapshots", colab=colab)
        Manager.make_directory(output_directory)
        current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        generated_image_file = f"{output_directory}/image[{current_time}].png"

        # Save the generated image at specified intervals and at the end.
        if save_interval is not None and (
            index % save_interval == 0 or index == inference_steps - 1
        ):
            plt.imsave(
                generated_image_file,
                np.array(np.clip((noise_image[0] + 1) * 127.5, 0, 255), np.uint8)[
                    :, :, 0
                ],
                cmap="gray",
            )

            # Print the status message.
            print(
                "\n🔽 "
                + Fore.BLUE
                + f"Generated picture {generated_image_file.split('/')[-1]} @ {output_directory}"
                + Style.RESET_ALL
            )

        # Display the generated image if verbose is True.
        if verbose:
            plt.imshow(
                np.array(
                    np.clip(
                        (np.squeeze(np.squeeze(noise_image, 0), -1) + 1) * 127.5, 0, 255
                    ),
                    np.uint8,
                ),
                cmap="gray",
            )
            plt.show()

    # Display the final generated image if verbose is True and the file name is assigned.
    if verbose and generated_image_file is not None:
        plt.imshow(
            np.array(np.clip((noise_image[0] + 1) * 127.5, 0, 255), np.uint8)[:, :, 0],
            cmap="gray",
        )
        plt.show()


def denoise_process(
    unet,
    denoising_method="DDPM",
    total_timesteps=100,
    inference_steps=10,
    starting_noise=None,
    verbose=True,
    save_interval=None,
    colab=0,
):

    if denoising_method == "DDPM":
        denoising_diffusion_probabilistic_models(
            unet,
            total_timesteps=total_timesteps,
            starting_noise=starting_noise,
            verbose=verbose,
            save_interval=save_interval,
            colab=colab,
        )

    elif denoising_method == "DDIM":
        denoising_diffusion_implicit_models(
            unet,
            total_timesteps=total_timesteps,
            inference_steps=inference_steps,
            starting_noise=starting_noise,
            verbose=verbose,
            save_interval=save_interval,
            colab=colab,
        )

    elif denoising_method == "BOTH":
        denoising_diffusion_probabilistic_models(
            unet,
            total_timesteps=total_timesteps,
            starting_noise=starting_noise,
            verbose=verbose,
            save_interval=save_interval,
            colab=colab,
        )

        denoising_diffusion_implicit_models(
            unet,
            total_timesteps=total_timesteps,
            inference_steps=inference_steps,
            starting_noise=starting_noise,
            verbose=verbose,
            save_interval=save_interval,
            colab=colab,
        )

    else:
        print("Invalid denoising method specified.")


# Example usage:
# denoise_process(unet, denoising_method="DDPM", timesteps=100, starting_noise=None, verbose=True, save_interval=None)
