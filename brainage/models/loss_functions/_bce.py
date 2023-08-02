"""BCE loss."""

# %% External package import

import torch.nn as nn

# %% Function definition


def BCELoss(x, y):
    """Return the binary cross entropy loss."""
    loss_func = nn.BCELoss()
    loss = loss_func(x, y)

    return loss
