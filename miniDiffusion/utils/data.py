from miniDiffusion.utils.preprocess import preprocess
from miniDiffusion.utils.params import BATCH_SIZE, data
import tensorflow_datasets as tfds
import tensorflow as tf
from colorama import Fore, Style

def get_datasets():
    # Load the MNIST dataset
    train_ds = tfds.load(data, as_supervised=True, split="train")

    # Normalize to [-1, 1], shuffle and batch
    train_ds = train_ds.map(preprocess, tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)

    print("\n⏹ " + Fore.GREEN +
          f"Data has been sucessfully preproccessed" +
          Style.RESET_ALL)

    print("\n🔽 " + Fore.GREEN +
          f"Data has been sucessfully loaded from {data} dataset" +
          Style.RESET_ALL)

    # Return numpy arrays instead of TF tensors while iterating
    return tfds.as_numpy(train_ds)
