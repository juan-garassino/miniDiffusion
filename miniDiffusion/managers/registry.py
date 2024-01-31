import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from miniDiffusion.utils.utils import forward_noise
from miniDiffusion.managers.manager import Manager
import os
from colorama import Fore, Style
from datetime import datetime

def some_steps(dataset, colab=0):
    # Let us visualize the output image at a few timestamps
    sample_mnist = next(iter(dataset))[0]

    out_dir = os.path.join(os.environ.get("HOME"), "Results", "miniDiffusion",
                           "snapshots")

    if int(colab) == 1:

        out_dir = os.path.join(os.environ.get("HOME"), "..", "content",
                               "results", "miniDiffusion", "snapshots")

    Manager.make_directory(out_dir)

    fig = plt.figure(figsize=(15, 30))

    for index, i in enumerate([10, 100, 150, 355]):
        noisy_im, noise = forward_noise(
            0,
            np.expand_dims(sample_mnist, 0),
            np.array(
                [
                    i,
                ]
            ),
        )
        plt.subplot(1, 4, index + 1)

        plt.imshow(np.squeeze(np.squeeze(noisy_im, -1), 0), cmap="gray")

        Manager.make_snapshot(fig, out_dir)

    plt.show()

# Save a GIF using logged images
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
