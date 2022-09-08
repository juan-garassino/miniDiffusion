from tensorflow.train import Checkpoint, CheckpointManager
import os
from colorama import Fore, Style

class Manager():

    def __init__(self, network):

        self.network = network
        self.checkpoint = Checkpoint(unet=self.network)
        self.directory = os.path.join(os.environ.get('HOME'), 'Results', 'miniDifussion',
                               'checkpoints')

        if int(os.environ.get('COLAB')) == 1:
            self.directory = os.path.join(os.environ.get('HOME'), '..', 'content',
                                          'results', 'miniDifussion',
                                          'checkpoints')

        self.checkpoint_manager = CheckpointManager(self.checkpoint, self.directory,
                                                    max_to_keep=2)

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
