"""Model performance evaluator."""

# %% Internal package import

from brainage.evaluation.metrics import (MeanSquaredError, MeanAbsoluteError,
                                         Correlation)

# %% Class definition


class ModelEvaluator():
    """
    Model performance evaluation class.

    This class provides ...

    Parameters
    ----------
    metrics : tuple
        ...

    Attributes
    ----------
    metrics : tuple
        See `Parameters`.

    metrics_selection : tuple
        ...

    Methods
    -------
    - ``select()`` : select the metrics from the catalogue;
    - ``compute_metrics(true_labels, predicted_labels)`` : compute all \
        selected metrics.
    """

    def __init__(
            self,
            metrics):

        print('\n\t Initializing the model evaluator ...')
        print('\t\t >>> Metrics: {} <<<'.format(metrics))

        # Get the attributes from the argument
        self.metrics = metrics

        # Construct the metrics list to compute
        self.metrics_selection = self.select()

    def select(self):
        """
        Select the metrics from the catalogue.

        Returns
        -------
        tuple
            ...
        """
        print('\t\t Selecting the metrics classes from the catalogue ...')

        return tuple(metric_class()
                     for metric_class in (Correlation, MeanAbsoluteError,
                                          MeanSquaredError)
                     for metric in self.metrics
                     if metric_class.label == metric)

    def compute_metrics(
            self,
            true_labels,
            predicted_labels):
        """
        Compute all selected metrics.

        Parameters
        ----------
        true_labels : ...
            ...

        predicted_labels : ...
            ...

        Returns
        -------
        metrics_dictionary : ...
            ...
        """
        # Compute the selected metrics and store the results as a dictionary
        metrics_dictionary = {metric.label: metric.compute(true_labels,
                                                           predicted_labels)
                              for metric in self.metrics_selection}

        return metrics_dictionary
