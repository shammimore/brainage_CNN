"""Calculate mean absolute error."""

# %% External package import

from sklearn.metrics import mean_absolute_error


# %% Class definition

class MeanAbsoluteError():
    """
    """
    
    label = 'MAE'

    def __init__(self):
        return
    
    def apply(self, true, predicted):
        return mean_absolute_error(true, predicted)
