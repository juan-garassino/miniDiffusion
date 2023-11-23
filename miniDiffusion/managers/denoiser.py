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
from miniDiffusion.utils.params import alpha, alpha_bar, beta, timesteps
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from miniDiffusion.managers.registry import save_gif
from miniDiffusion.managers.manager import Manager
import os
from datetime import datetime
from colorama import Fore, Style

def ddpm(x_t, pred_noise, t):
    # alpha_t and alpha_t_bar are extracted from pre-defined sequences alpha and alpha_bar
    # using the timestep t. These values are part of the noise schedule in the diffusion process.
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    # Calculate the coefficient for scaling the predicted noise.
    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5

    # Compute the mean of the reverse diffusion process at timestep t.
    # This is part of the denoising step, reconstructing the data from the noisy version.
    mean = (1 / (alpha_t**0.5)) * (x_t - eps_coef * pred_noise)

    # Variance at timestep t is obtained from the pre-defined sequence beta.
    var = np.take(beta, t)

    # Sample noise from a normal distribution to add stochasticity to the reverse process.
    z = np.random.normal(size=x_t.shape)

    # Return the denoised data with added noise according to the variance at timestep t.
    return mean + (var**0.5) * z

def ddim(x_t, pred_noise, t, sigma_t):
    # Extract alpha_t_bar and alpha_t_minus_one using timestep t from pre-defined sequences.
    alpha_t_bar = np.take(alpha_bar, t)
    alpha_t_minus_one = np.take(alpha, t - 1)

    # Compute the prediction by denoising and adjusting for the alpha coefficients.
    pred = (x_t - ((1 - alpha_t_bar) ** 0.5) * pred_noise) / (alpha_t_bar**0.5)
    pred = (alpha_t_minus_one**0.5) * pred

    # Adjust the prediction by adding a scaled version of the predicted noise.
    pred = pred + ((1 - alpha_t_minus_one - (sigma_t**2)) ** 0.5) * pred_noise

    # Add additional noise scaled by sigma_t for stochasticity.
    eps_t = np.random.normal(size=x_t.shape)
    pred = pred + (sigma_t * eps_t)

    # Return the final prediction for the timestep t.
    return pred


def denoising_diffusion_probabilistic_models(unet):
    # Starting the Denoising Diffusion Probabilistic Model process.
    print("\n⏹ " + Fore.GREEN + "Denoising Diffusion Probabilistic Model started" + Style.RESET_ALL)

    # Initialize a random noise image.
    x = tf.random.normal((1, 32, 32, 1))
    img_list = [np.squeeze(np.squeeze(x, 0), -1)]

    # Iteratively apply the denoising diffusion process.
    for i in range(timesteps - 1):
        # Calculate the current timestep.
        t = np.expand_dims(np.array(timesteps - i - 1, np.int32), 0)
        # Predict the noise using the U-Net model.
        pred_noise = unet(x, t)
        # Apply the denoising diffusion process.
        x = ddpm(x, pred_noise, t)
        # Store the generated image for later use.
        img_list.append(np.squeeze(np.squeeze(x, 0), -1))

        # Every 25 steps, save and display the current generated image.
        if i % 25 == 0:
            # Display the generated image.
            plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255)[:, :, 0], np.uint8), cmap="gray")

            # Define the directory and filename for saving the image.
            out_dir = Manager.working_directory('snapshots')
            Manager.make_directory(out_dir)
            now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            picture_name = f"{out_dir}/image[{now}][{i}].png"

            # Print the status message and show the image.
            print("\n🔽 " + Fore.BLUE + f"Generated picture {picture_name.split('/')[-1]} @ {out_dir}" + "\n" + Style.RESET_ALL)
            plt.show()

    # Generate and save the final animation as a GIF.
    save_gif(img_list + ([img_list[-1]] * 100), picture_name, interval=20)

    # Display and print the final generated image.
    plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255)[:, :, 0], np.uint8))
    print("\n🔽 " + Fore.BLUE + f"Generated git {picture_name.split('/')[-1]} at {out_dir}" + "\n" + Style.RESET_ALL)
    plt.show()


def denoising_diffusion_implicit_models(unet):
    # Starting the Denoising Diffusion Implicit Model process.
    print("\n⏹ " + Fore.GREEN + "Denoising Diffusion Implicit Model started" + Style.RESET_ALL)

    # Define the number of inference steps.
    inference_timesteps = 10
    inference_range = range(0, timesteps, timesteps // inference_timesteps)

    # Initialize a random noise image.
    x = tf.random.normal((1, 32, 32, 1))
    img_list = [np.squeeze(np.squeeze(x, 0), -1)]

    # Iteratively apply the denoising diffusion implicit process.
    for index, i in tqdm(enumerate(reversed(range(inference_timesteps))), total=inference_timesteps):
        t = np.expand_dims(inference_range[i], 0)
        pred_noise = unet(x, t)
        x = ddim(x, pred_noise, t, 0)
        img_list.append(np.squeeze(np.squeeze(x, 0), -1))

        # Display the generated image at each step.
        if index % 1 == 0:
            plt.imshow(np.array(np.clip((np.squeeze(np.squeeze(x, 0), -1) + 1) * 127.5, 0, 255), np.uint8), cmap="gray")

            # Define the directory and filename for saving the image.
            out_dir = Manager.working_directory('snapshots')
            Manager.make_directory(out_dir)
            now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            picture_name = "{}/image[{}].png".format(out_dir, now)

            # Print the status message and show the image.
            print("\n🔽 " + Fore.BLUE + f"Generated picture {picture_name.split('/')[-1]} @ {out_dir}" + "\n" + Style.RESET_ALL)
            plt.show()

    # Save the final image.
    plt.savefig(picture_name)

    # Display the final generated image.
    plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255), np.uint8)[:, :, 0], cmap="gray")
    print("\n🔽 " + Fore.BLUE + f"Generated picture {picture_name.split('/')[-1]} @ {out_dir}" + Style.RESET_ALL)
    plt.show()
