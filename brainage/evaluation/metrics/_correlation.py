"""Correlation computation."""

# %% External package import

from numpy import corrcoef

# %% Class definition


class Correlation():
    """
    Correlation computation class.

    This class provides ...

    Attributes
    ----------
    ...

    Methods
    -------
    ...
    """

    label = 'CORR'

    def __init__(self):
        return

    def compute(
            self,
            true_labels,
            predicted_labels):
        """Compute the correlation."""
        return corrcoef(true_labels, predicted_labels)[0, 1]
