""" Plot results. """

# %% Internal package import

# %% External package import

import matplotlib.pyplot as plt
from pathlib import Path

# %% Class definition


class TrainingLossPlot():
    """
    Training loss class.

    This class provides ...

    Attributes
    ----------
    label : string
        ...

    Methods
    -------
    - ``view(tracker, save_path)`` : plots training loss vs number of epochs\
    and saves it
    """
        
    label = 'Training_loss'

    def __init__(self, tracker, save_path):
        self.tracker = tracker
        self.save_path =  save_path

    def view(self):
        plt.figure()
        plt.scatter(range(len(self.tracker['training_loss'])), 
                    self.tracker['training_loss'])
        plt.gca().set(xlabel='Epochs', ylabel='Training loss')
        plt.savefig(Path(self.save_path, 'training_loss.png'))

