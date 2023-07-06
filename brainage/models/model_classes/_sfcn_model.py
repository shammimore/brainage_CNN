"""SFCN model."""

# %% External package import

import torch.nn as nn

# %% Internal package import

from brainage.models.architectures import SFCN

# %% Class definition


class SFCNModel(nn.Module):
    """
    SFCN model class.

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
            pretrained_weights,
            device):

        super(SFCNModel, self).__init__()
        self.architecture = SFCN()
        self.architecture = nn.DataParallel(self.architecture)

    def forward(
            self,
            image):
        """Perform a forward pass through the model."""
        out = self.architecture(image)
        return out
