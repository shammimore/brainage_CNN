"""Mean absolute error computation."""

# %% External package import

from sklearn.metrics import mean_absolute_error

# %% Class definition


class MeanAbsoluteError():
    """
    Mean absolute error computation class.

    This class provides ...

    Attribute
    ---------
    ...

    Methods
    -------
    ...
    """

    label = 'MAE'

    def __init__(self):
        return

    def compute(
            self,
            true_labels,
            predicted_labels):
        """Compute the mean absolute error."""
        return mean_absolute_error(true_labels, predicted_labels)
