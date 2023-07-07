"""Model performance evaluator."""

# %% External package import


# %% Internal package import

from brainage.evaluation.metrics import MeanSquaredError, MeanAbsoluteError, Correlation

# %% Class definition

class ModelEvaluator():
    """

    """

    def __init__(
            self, 
            metrics=()):
        
        # Get the attributes from the argument
        self.metrics = metrics

        # Specify the performance metrics catalogue
        self.metrics_catalogue = (MeanSquaredError, MeanAbsoluteError, Correlation)

        # Build the preprocessing pipeline
        self.pipeline =  self.build()

    
    def build(self):
        """
        """
        return tuple(metric_class()
                     for metric_class in self.metrics_catalogue 
                     for metric in self.metrics 
                     if metric_class.label == metric )
    
    def run_metrics(
            self,
            true, predicted):
        """
        Run all metrics.

        Parameters
        ----------
        image : ...
            ...

        Returns
        -------
        image_data : ...
            ...
        """

        # Run the metric steps sequentially and store results in a dictionary
        performance_metric = {}
        for step in self.pipeline:
            value = step.apply(true, predicted)
            performance_metric[step.label] = value
            
        return performance_metric









    

# %%
