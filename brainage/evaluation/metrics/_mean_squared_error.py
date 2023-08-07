"""Mean squared error."""

# %% External package import

from sklearn.metrics import mean_squared_error

# %% Class definition


class MeanSquaredError():
    """
    Mean squared error class.

    This class provides ...

    Attributes
    ----------
    label : string
        ...

    Methods
    -------
    - ``compute(true_labels, predicted_labels)`` : compute the mean squared \
        error.
    """

    label = 'MSE'

    def compute(
            self,
            true_labels,
            predicted_labels):
        """
        Compute the mean squared error.

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
        return mean_squared_error(true_labels, predicted_labels)
