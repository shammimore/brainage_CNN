"""Correlation between true and predicted."""

# %% External package import

from numpy import corrcoef


# %% Class definition

class Correlation():
    """
    """

    label = 'CoRR'

    def __init__(self):
        return
    
    def apply(self, true, predicted):
        return corrcoef(true, predicted)[0,1]
