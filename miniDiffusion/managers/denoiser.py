import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from miniDiffusion.utils.params import alpha, alpha_bar, timesteps, beta
from miniDiffusion.managers.registry import save_gif

# Denoising Diffusion Probabilistic Models (DDPM)


def ddpm(x_t, pred_noise, t):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5
    mean = (1 / (alpha_t**0.5)) * (x_t - eps_coef * pred_noise)

    var = np.take(beta, t)
    z = np.random.normal(size=x_t.shape)

    return mean + (var**0.5) * z


# Denoising Diffusion Implicit Models (DDIM)


def ddim(x_t, pred_noise, t, sigma_t):
    alpha_t_bar = np.take(alpha_bar, t)
    alpha_t_minus_one = np.take(alpha, t - 1)

    pred = (x_t - ((1 - alpha_t_bar) ** 0.5) * pred_noise) / (alpha_t_bar**0.5)
    pred = (alpha_t_minus_one**0.5) * pred

    pred = pred + ((1 - alpha_t_minus_one - (sigma_t**2)) ** 0.5) * pred_noise
    eps_t = np.random.normal(size=x_t.shape)
    pred = pred + (sigma_t * eps_t)

    return pred


if __name__ == "__main__":
    if os.environ["DENOISING"] == "ddpm":

        x = tf.random.normal((1, 32, 32, 1))
        img_list = []
        img_list.append(np.squeeze(np.squeeze(x, 0), -1))

        for i in tqdm(range(timesteps - 1)):
            t = np.expand_dims(np.array(timesteps - i - 1, np.int32), 0)
            pred_noise = unet(x, t)
            x = ddpm(x, pred_noise, t)
            img_list.append(np.squeeze(np.squeeze(x, 0), -1))

            if i % 25 == 0:
                plt.imshow(np.array(
                    np.clip((x[0] + 1) * 127.5, 0, 255)[:, :, 0], np.uint8),
                           cmap="gray")
                plt.show()

        save_gif(img_list + ([img_list[-1]] * 100), "ddpm.gif", interval=20)

        plt.imshow(
            np.array(np.clip((x[0] + 1) * 127.5, 0, 255)[:, :, 0], np.uint8))
        plt.show()

    if os.environ["DENOISING"] == "ddim":
        # Define number of inference loops to run
        inference_timesteps = 10

        # Create a range of inference steps that the output should be sampled at
        inference_range = range(0, timesteps, timesteps // inference_timesteps)

        x = tf.random.normal((1, 32, 32, 1))
        img_list = []
        img_list.append(np.squeeze(np.squeeze(x, 0), -1))

        # Iterate over inference_timesteps
        for index, i in tqdm(
            enumerate(reversed(range(inference_timesteps))), total=inference_timesteps
        ):
            t = np.expand_dims(inference_range[i], 0)

            pred_noise = unet(x, t)

            x = ddim(x, pred_noise, t, 0)
            img_list.append(np.squeeze(np.squeeze(x, 0), -1))

            if index % 1 == 0:
                plt.imshow(
                    np.array(
                        np.clip((np.squeeze(np.squeeze(x, 0), -1) + 1) * 127.5, 0, 255),
                        np.uint8,
                    ),
                    cmap="gray",
                )
                plt.show()

        plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255),
                            np.uint8)[:, :, 0],
                   cmap="gray")
        plt.show()
