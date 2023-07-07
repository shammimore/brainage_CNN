"""Calculate mean squared error."""

# %% External package import

from sklearn.metrics import mean_squared_error


# %% Class definition

class MeanSquaredError():
    """
    """
    
    label = 'MSE'

    def __init__(self):
        return 
    
    def apply(self, true, predicted):
        return mean_squared_error(true, predicted)

