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
    label : string
        ...

    Methods
    -------
    - ``compute(true_labels, predicted_labels)`` : compute the mean absolute \
        error.
    """

    label = 'MAE'

    def compute(
            self,
            true_labels,
            predicted_labels):
        """
        Compute the mean absolute error.

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
        return mean_absolute_error(true_labels, predicted_labels)
