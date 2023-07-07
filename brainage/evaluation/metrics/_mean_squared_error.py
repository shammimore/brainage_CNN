"""Mean squared error computation."""

# %% External package import

from sklearn.metrics import mean_squared_error

# %% Class definition


class MeanSquaredError():
    """
    Mean squared error computation class.

    This class provides ...

    Attributes
    ----------
    ...

    Methods
    -------
    ...
    """

    label = 'MSE'

    def __init__(self):
        return

    def compute(
            self,
            true_labels,
            predicted_labels):
        """Compute the mean squared error."""
        return mean_squared_error(true_labels, predicted_labels)
