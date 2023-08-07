"""Evaluation metrics module."""

from ._correlation import Correlation
from ._mean_absolute_error import MeanAbsoluteError
from ._mean_squared_error import MeanSquaredError

__all__ = ['Correlation',
           'MeanAbsoluteError',
           'MeanSquaredError']
