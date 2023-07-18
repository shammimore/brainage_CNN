"""Visualization."""

# %% External package import


# %% Internal package import

from brainage.visualization.plots import (TrainingLossPlot,
                                          ValidationLossPlot)

# %%


class Visualizer():
    """
    Visualization class.

    This class provides ...

    Parameters
    ----------
    ...

    Attributes
    ----------
    ...

    Methods
    -------
    ...
    """

    def __init__(
            self,
            tracker,
            save_path):

        print('\n\t Initializing the visualizer ...')

        # Get the tracking dictionary from the argument
        self.tracker = tracker
        self.save_path = save_path

        # Create a mapping between plot labels and classes
        self.plot_catalogue = {TrainingLossPlot.label: TrainingLossPlot,
                               ValidationLossPlot.label: ValidationLossPlot}
        

    def open_plot(
            self,
            name):
        """."""
        # Initialize the plotting class
        plotter = self.plot_catalogue[name](self.tracker, self.save_path)

        # View the plot
        plotter.view()
