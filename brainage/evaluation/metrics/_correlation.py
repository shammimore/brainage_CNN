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
    label : string
        ...

    Methods
    -------
    - ``compute(true_labels, predicted_labels)`` : compute the correlation.
    """

    label = 'CORR'

    def compute(
            self,
            true_labels,
            predicted_labels):
        """
        Compute the correlation.

        Parameters
        ----------
        true_labels : ...
            ...

        predicted_labels : ...
            ...

        Returns
        -------
        ...
        """
        return corrcoef(true_labels, predicted_labels)[0, 1]
