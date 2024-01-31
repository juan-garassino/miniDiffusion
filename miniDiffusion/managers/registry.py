import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from miniDiffusion.managers.manager import Manager
import os
from colorama import Fore, Style
import imageio
from miniDiffusion.utils.utils import forward_noise

def some_steps(dataset, colab=0, steps_range=100):
    # Let us visualize the output image at a few timestamps
    sample_mnist = next(iter(dataset))[0]

    # Define the output directory
    out_dir = os.path.join(os.environ.get("HOME"), "Code", "juan-garassino", "miniDiffusion" , "miniDiffusion", "Results", "miniDiffusion", "snapshots")

    # Adjust the output directory if running on Colab
    if int(colab) == 1:
        out_dir = os.path.join(os.environ.get("HOME"), "..", "content", "results", "miniDiffusion", "snapshots")

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over the specified range of steps
    for i in range(steps_range):

        print(i)
        # Generate noisy image and noise
        noisy_im, noise = forward_noise(
            0,
            np.expand_dims(sample_mnist, 0),
            np.array([i]),
        )

        # Create a new figure for each step
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image, added noise, and noisy image
        axes[0].imshow(sample_mnist, cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(np.squeeze(noise), cmap="gray")
        axes[1].set_title("Added Noise")

        axes[2].imshow(np.squeeze(noisy_im), cmap="gray")
        axes[2].set_title("Noisy Image")

        # Save plot to snapshot directory
        snapshot_path = os.path.join(out_dir, f"snapshot_{i}.png")
        plt.savefig(snapshot_path)
        plt.close()

def save_gif_from_images(input_path, output_path, interval=200):
    # Get a list of image file names in the input directory
    image_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    # Sort image files based on their names
    image_files.sort()

    print(image_files)

    # Load images from the input directory
    imgs = []

    for file_name in image_files:
        file_path = os.path.join(input_path, file_name)
        img = Image.open(file_path)
        imgs.append(img)

    # Save images as GIF
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

    # Load MNIST dataset
    manager = Manager()  # Initialize the project manager

    dataset = manager.get_datasets(dataset='mnist', samples=50000, batch_size=64)  # Project manager loads the data

    some_steps(dataset, colab=0, steps_range=10)

    input_path='/Users/juan-garassino/Code/juan-garassino/miniDiffusion/miniDiffusion/Results/miniDiffusion/snapshots'

    output_path='/Users/juan-garassino/Code/juan-garassino/miniDiffusion/miniDiffusion/Results'

    save_gif_from_images(input_path, output_path, interval=200)

    # # Get the first sample from the dataset
    # first_sample = next(iter(mnist_dataset))[0]

    # # Define the output directory
    # out_dir = os.path.join(os.getcwd(), "Results", "miniDiffusion", "snapshots")

    # # Create the noising GIF
    # gif_path = create_noising_gif(first_sample, colab=0, out_dir=out_dir)

    # # Display the path to the generated GIF
    # print("\n🔽 " + Fore.BLUE + f"Generated GIF: {gif_path}" + Style.RESET_ALL)
