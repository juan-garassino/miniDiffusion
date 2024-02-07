# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from miniDiffusion.managers.manager import Manager
# import os
# from colorama import Fore, Style
# import imageio
# from miniDiffusion.utils.utils import forward_noise

# def some_steps(dataset, colab=0, steps_range=100):
#     # Let us visualize the output image at a few timestamps
#     sample_mnist = next(iter(dataset))[0]

#     # Define the output directory
#     out_dir = os.path.join(os.environ.get("HOME"), "Code", "juan-garassino", "miniDiffusion" , "miniDiffusion", "Results", "miniDiffusion", "snapshots")

#     # Adjust the output directory if running on Colab
#     if int(colab) == 1:
#         out_dir = os.path.join(os.environ.get("HOME"), "..", "content", "results", "miniDiffusion", "snapshots")

#     # Create the output directory if it doesn't exist
#     os.makedirs(out_dir, exist_ok=True)

#     # Iterate over the specified range of steps
#     for i in range(steps_range):

#         rng, tsrng = np.random.randint(0, 100000, size=(2,))

#         # print(i)

#         # Generate noisy image and noise
#         noisy_im, noise = forward_noise(
#             rng,
#             np.expand_dims(sample_mnist, 0),
#             np.array([i]),
#         )

#         # Create a new figure for each step
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#         # Plot original image, added noise, and noisy image
#         axes[0].imshow(sample_mnist, cmap="gray")
#         axes[0].set_title("Original")

#         axes[1].imshow(np.squeeze(noise), cmap="gray")
#         axes[1].set_title("Added Noise")

#         axes[2].imshow(np.squeeze(noisy_im), cmap="gray")
#         axes[2].set_title("Noisy Image")

#         # Save plot to snapshot directory
#         snapshot_path = os.path.join(out_dir, f"snapshot_{i:04d}.png")
#         plt.savefig(snapshot_path)
#         plt.close()

# def save_gif(img_list, path="", interval=200):
#     # Transform images from [-1,1] to [0, 255]
#     imgs = []
#     for im in img_list:
#         im = np.array(im)
#         im = (im + 1) * 127.5
#         im = np.clip(im, 0, 255).astype(np.int32)
#         im = Image.fromarray(im)
#         imgs.append(im)

#     imgs = iter(imgs)

#     # Extract first image from iterator
#     img = next(imgs)

#     # Append the other images and save as GIF
#     img.save(
#         fp=path,
#         format="GIF",
#         append_images=imgs,
#         save_all=True,
#         duration=interval,
#         loop=0,
#     )

#     print("\n🔽 " + Fore.BLUE +
#           "Genearted gif of {} frames from {}".format(interval, path) +
#           Style.RESET_ALL)

# def gif_from_directoty(input_path, output_path, interval=200):
#     # Get a list of image file names in the input directory
#     image_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

#     # Sort image files based on their names
#     image_files.sort()

#     # Load images from the input directory
#     imgs = []

#     for file_name in image_files:
#         file_path = os.path.join(input_path, file_name)
#         img = Image.open(file_path)
#         img.load()  # Ensure image data is fully loaded
#         imgs.append(img)

#     # Save images as GIF
#     if not os.path.isdir(output_path):  # Check if output_path is a directory
#         imgs[0].save(
#             fp=output_path,
#             format="GIF",
#             append_images=imgs[1:],
#             save_all=True,
#             duration=interval,
#             loop=0
#         )

#         print("\n🔽 " + Fore.BLUE +
#               f"Generated GIF of {len(imgs)} frames from {input_path} to {output_path}" +
#               Style.RESET_ALL)
#     else:
#         print("\n🔺 " + Fore.RED +
#               f"Error: Output path {output_path} is a directory. Provide a valid file path." +
#               Style.RESET_ALL)

# if __name__ == "__main__":

#     from miniDiffusion.managers.manager import Manager

#     manager = Manager()  # Initialize the project manager

#     dataset = manager.get_datasets(dataset='mnist', samples=1000, batch_size=1)  # Project manager loads the data

#     some_steps(dataset, colab=0, steps_range=500)

#     input_path='/Users/juan-garassino/Code/juan-garassino/miniDiffusion/miniDiffusion/Results/miniDiffusion/snapshots'

#     output_path='/Users/juan-garassino/Code/juan-garassino/miniDiffusion/miniDiffusion/Results/animation.gif'

#     gif_from_directoty(input_path, output_path, interval=200)
