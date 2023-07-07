"""Model performance evaluator."""

# %% External package import

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
            metrics=()):

        # Get the attributes from the argument
        self.metrics = metrics

        # Specify the performance metrics catalogue
        self.metrics_catalogue = (MeanSquaredError, MeanAbsoluteError,
                                  Correlation)

        # Construct the metrics list to compute
        self.metrics_selection = self.select()

    def select(self):
        """Select the metrics from the catalogue."""
        return tuple(metric_class()
                     for metric_class in self.metrics_catalogue
                     for metric in self.metrics
                     if metric_class.label == metric)

    def compute_metrics(
            self,
            true_labels,
            predicted_labels):
        """
        Compute all metrics.

        Parameters
        ----------
        image : ...
            ...

        Returns
        -------
        image_data : ...
            ...
        """
        # Compute the selected metrics and store the results as a dictionary
        performance_metric = {metric.label: metric.compute(true_labels,
                                                           predicted_labels)
                              for metric in self.metrics_selection}

        return performance_metric
