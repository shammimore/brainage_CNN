"""Loss functions module."""

from ._bce import BCELoss
from ._kld import KLDivLoss

__all__ = ['BCELoss',
           'KLDivLoss']
