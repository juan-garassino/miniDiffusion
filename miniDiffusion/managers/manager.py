from tensorflow.train import Checkpoint, CheckpointManager
import os
from colorama import Fore, Style

from miniDiffusion.utils.preprocess import preprocess
from miniDiffusion.utils.params import BATCH_SIZE

import tensorflow_datasets as tensorflow_datasets
import tensorflow as tf

class Manager():

    def __init__(self, network, data):

        self.network = network
        self.checkpoint = Checkpoint(unet=self.network)
        self.directory = os.path.join(os.environ.get('HOME'), 'Results', 'miniDifussion',
                               'checkpoints')

        self.make_directory(self.directory)

        if int(os.environ.get('COLAB')) == 1:
            self.directory = os.path.join(os.environ.get('HOME'), '..', 'content',
                                          'results', 'miniDifussion',
                                          'checkpoints')

        self.checkpoint_manager = CheckpointManager(self.checkpoint, self.directory,
                                                    max_to_keep=2)

        self.data = data

    def load_model(self):
        # load from a previous checkpoint if it exists, else initialize the model from scratch
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            start_interation = int(self.checkpoint_manager.latest_checkpoint.split("-")[-1])

            print("\n🔽 " + Fore.BLUE + "Restored from ckeckpoint{}".format(
                start_interation) +
                  Style.RESET_ALL)

        else:

            print("\n⏹ " + Fore.GREEN + "Initializing from scratch." + Style.RESET_ALL)

        return self.checkpoint, self.checkpoint_manager

    def get_datasets(self):
        # Load the MNIST dataset
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        train_ds = tensorflow_datasets.load(name=os.environ.get('DATA'),
                                            data_dir=data_dir,
                                            shuffle_files=True,
                                            as_supervised=True,
                                            split=['train', 'test'][0],
                                            with_info=True,
                                            download=True)

        # Normalize to [-1, 1], shuffle and batch
        train_ds = train_ds[0].map(preprocess, tf.data.AUTOTUNE)

        train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE)

        print("\n⏹ " + Fore.GREEN +
            f"Data has been sucessfully preproccessed" +
            Style.RESET_ALL)

        print(
            "\n🔽 " + Fore.GREEN +
            f"Data has been sucessfully loaded from {os.environ.get('DATA')} dataset"
            + Style.RESET_ALL)

        # Return numpy arrays instead of TF tensors while iterating
        return tensorflow_datasets.as_numpy(train_ds)

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print("\n⏹ " + Fore.GREEN +
                  f"This directory has been created {directory}" +
                  Style.RESET_ALL)

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
