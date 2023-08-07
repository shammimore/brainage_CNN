"""BCE loss."""

# %% External package import

from torch.nn import BCELoss as TorchBCELoss

# %% Function definition


def BCELoss(prediction, ground_truth):
    """
    Compute the binary cross-entropy (BCE) loss.

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
    """
    loss_func = TorchBCELoss()
    loss = loss_func(prediction, ground_truth)

    return loss
