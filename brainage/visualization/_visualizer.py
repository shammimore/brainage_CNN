"""Visualizer."""

# %% Internal package import

from brainage.visualization.plots import (TrainingLossPlot,
                                          ValidationLossPlot)

# %% Class definition


class Visualizer():
    """
    Visualizer class.

    This class provides ...

    Parameters
    ----------
    tracker : dict
        ...

    save_path : string
        ...

    Attributes
    ----------
    tracker : dict
        See 'Parameters'.

    save_path : string
        See 'Parameters'.

    plot_catalogue : dict
        ...

    Methods
    -------
    - ``open_plot(name)`` : initialize and view a plot.
    """

    def __init__(
            self,
            tracker,
            save_path):

        print('\n\t Initializing the visualizer ...')

        # Get the attributes from the argument
        self.tracker = tracker
        self.save_path = save_path

        # Get all plotting classes
        plotters = (TrainingLossPlot, ValidationLossPlot)

        # Create a mapping between plot labels and classes
        self.plot_catalogue = {plotter.label: plotter for plotter in plotters}

    def open_plot(
            self,
            name):
        """Initialize and view a plot."""
        # Initialize the plotting class
        plotter = self.plot_catalogue[name](self.tracker, self.save_path)

        # View the plot
        plotter.view()
