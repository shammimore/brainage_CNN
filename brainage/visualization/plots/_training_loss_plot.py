"""Training loss plot."""

# %% External package import

from matplotlib.pyplot import savefig, subplots
from numpy import linspace
from pathlib import Path
from seaborn import regplot

# %% Class definition


class TrainingLossPlot():
    """
    Training loss plot class.

    This class provides ...

    Parameters
    ----------
    tracker : ...
        ...

    save_path : ...
        ...

    Attributes
    ----------
    label : string
        ...

    tracker : ...
        See 'Parameters'.

    save_path : ...
        See 'Parameters'.

    Methods
    -------
    - ``view()`` : plot the training loss per epoch.
    """

    label = 'Training_loss'

    def __init__(
            self,
            tracker,
            save_path):

        # Get the attributes from the arguments
        self.tracker = tracker
        self.save_path = save_path

    def view(self):
        """Plot the training loss per epoch."""
        # Create a new figure and axis
        figure, axis = subplots(nrows=1, figsize=(8, 5))

        # Add the scatter/regression plot
        regplot(x=linspace(1, len(self.tracker['training_loss']),
                           len(self.tracker['training_loss'])),
                y=self.tracker['training_loss'], lowess=True,
                line_kws={"color": "orange"})

        # Set the plot title
        axis.set_title('Training loss per epoch', fontsize=14,
                       fontweight='bold')

        # Set the x- and y-label
        axis.set_xlabel('Epochs', fontsize=12)
        axis.set_ylabel('Training loss', fontsize=12)

        # Set the x-limits
        axis.set_xlim((0, len(self.tracker['training_loss'])+1))

        # Set the facecolor for the axis
        axis.set_facecolor("whitesmoke")

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(which='minor', color='lightgray', linestyle=':',
                  linewidth=0.5)
        axis.minorticks_on()

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Save the plot
        savefig(Path(self.save_path, 'training_loss.png'))
