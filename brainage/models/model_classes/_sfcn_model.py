"""SFCN model."""

# %% External package import
import torch
import numpy as np

# %% Internal package import
from brainage.models.architectures import SFCN
from brainage.tools import random_seed, num2vect

# %% Class definition


class SFCNModel(torch.nn.Module):
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
        self.architecture = torch.nn.DataParallel(self.architecture)
        self.device = device
        
        if pretrained_weights:
            self.architecture.load_state_dict(torch.load(pretrained_weights, map_location=torch.device(device)))

    def forward(
            self,
            image, label):
        """Perform a forward pass through the model."""
        
        bin_step = 1
        sigma = 1
        age_range = [42, 82]
        y, bc = num2vect(label, age_range, bin_step, sigma)
        c, d, h, w = image.shape
        b = 1 # batch size would be one here
        image = image.reshape(1, 1, d, h, w)
        image = torch.as_tensor(image, dtype=torch.float32, device=self.device)

        self.architecture.eval()  # Don't forget this. BatchNorm will be affected if not in eval mode.
        with torch.no_grad():
            output = self.architecture(image)
        out = output[0].cpu().reshape([b, -1])  # bring it back to cpu
        out = out.numpy()
        prob = np.exp(out)
        pred = prob@bc
        print('pred:', pred)

        return pred
