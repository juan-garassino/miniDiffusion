from tensorflow.train import Checkpoint, CheckpointManager
import os
from colorama import Fore, Style
import errno
import matplotlib.pyplot as plt
from datetime import datetime

from miniDiffusion.tools.preprocess import preprocess
from miniDiffusion.tools.params import BATCH_SIZE, SAMPLES

import tensorflow_datasets as tensorflow_datasets
import tensorflow as tf


class Manager:
    def __init__(self):  # , network, optimizer, data):
        pass
        # self.data = data

    @staticmethod  # They do not require a class instance creation
    def train_and_checkpoint(ckeckpoint, manager):
        ckeckpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(
                "\n✅ "
                + Fore.CYAN
                + "Restored from {}".format(manager.latest_checkpoint)
                + Style.RESET_ALL
            )
        else:
            print("\n✅ " + Fore.CYAN + "Initializing from scratch." + Style.RESET_ALL)

    @staticmethod  # They do not require a class instance creation
    def working_directory(subdirectory, colab=0):
        directory = os.path.join(
            os.environ.get("HOME"), "Results", "miniDiffusion", subdirectory
        )

        if int(colab) == 1:
            directory = os.path.join(
                os.environ.get("HOME"),
                "..",
                "content",
                "results",
                "miniDiffusion",
                subdirectory,
            )

        return directory

    @staticmethod  # They do not require a class instance creation
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print(
                "\n⏹ "
                + Fore.GREEN
                + f"This directory has been created {directory}"
                + Style.RESET_ALL
            )

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod  # They do not require a class instance creation
    def get_datasets(dataset="fashion_mnist", samples=1000, batch_size=64):
        # Load the MNIST dataset
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

        train_ds = tensorflow_datasets.load(
            name=dataset,
            data_dir=data_dir,
            shuffle_files=True,
            as_supervised=True,
            split=["train", "test"][0],
            with_info=True,
            download=True,
        )

        # Normalize to [-1, 1], shuffle and batch
        train_ds = train_ds[0].map(preprocess, tf.data.AUTOTUNE)

        train_ds = (
            train_ds.take(samples)
            .shuffle(5000)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )

        print(
            "\n⏹ "
            + Fore.GREEN
            + f"Data has been sucessfully preproccessed"
            + Style.RESET_ALL
        )

        print(
            "\n🔽 "
            + Fore.GREEN
            + f"Data has been sucessfully loaded from {dataset} dataset"
            + Style.RESET_ALL
        )

        # Return numpy arrays instead of TF tensors while iterating
        return tensorflow_datasets.as_numpy(train_ds)

    @staticmethod  # They do not require a class instance creation
    def make_snapshot_label(output_directory):

        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        picture_name = "{}/animation[{}].png".format(
            output_directory, now
        )  # BE CAREFULL WITH GIF!!

        return picture_name

    @staticmethod  # They do not require a class instance creation
    def make_snapshot(snapshot, out_dir):

        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        picture_name = "{}/image[{}].png".format(out_dir, now)

        plt.savefig(picture_name)

        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Generated media {picture_name.split('/')[-1]} at {out_dir}"
            + Style.RESET_ALL
        )
