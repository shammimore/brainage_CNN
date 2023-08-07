"""Kullback-Leibler divergence loss."""

# %% External package import

from torch.nn import KLDivLoss as TorchKLDivLoss

# %% Function definition


def KLDivLoss(prediction, ground_truth):
    """
    Compute the Kullback-Leibler divergence loss.

    Parameters
    ----------
    prediction : ...
        ...

    ground_truth : ...
        ...

    Returns
    -------
    loss : ...
        ...

    Notes
    -----
    This implementation is different from the default PyTorch `KLDivLoss` in \
    that a) the resulting loss is averaged by the 0th dimension (batch size), \
    and b) a small value (1e-16) is added to the distribution of the ground \
    truth values to prevent numerical problems with log(0).
    """
    loss_func = TorchKLDivLoss(reduction='sum')
    loss = loss_func(prediction, ground_truth+1e-16) / ground_truth.shape[0]

    return loss
