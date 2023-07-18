""" Plot results. """

# %% Internal package import

# %% External package import

import matplotlib.pyplot as plt
from pathlib import Path

# %% Class definition


class ValidationLossPlot():
    """
    Validation loss class.

    This class provides ...

    Attributes
    ----------
    label : string
        ...

    Methods
    -------
    - ``view(tracker, save_path)`` : plots validation loss vs number of epochs\
    and saves it
    """
        
    label = 'Validation_loss'

    def __init__(self, tracker, save_path):
        self.tracker = tracker
        self.save_path =  save_path

    def view(self):
        plt.figure()
        plt.scatter(range(len(self.tracker['validation_loss'])), 
                    self.tracker['validation_loss'])
        plt.gca().set(xlabel='Epochs', ylabel='Validation loss')
        plt.savefig(Path(self.save_path, 'validation_loss.png'))



