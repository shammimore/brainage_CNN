"""Data model predictor."""

# %% External package import

import torch
import random
import numpy as np

# %% Internal package import

from brainage.models.architectures import SFCN

# %% Class definition


def random_seed(seed_value, device):
    """Set the random seed."""
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if device == "cuda:0":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True   # needed
        torch.backends.cudnn.benchmark = False


class DataModelPredictor():
    """
    Data model predictor class.

    This class provides ...

    Parameters
    ----------
    X

    Attributes
    ----------
    X

    Methods
    -------
    X
    """

    def __init__(
            self,
            data_loader,
            data_preprocessor,
            learning_rate,
            number_of_epochs,
            batch_size,
            train_all_layers,
            architecture,
            optimizer,
            pretrained_weights):

        # Get attributes from the arguments
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        self.architecture = architecture

        # Get the csv data
        self.raw_data = self.data_loader.get_data('raw')

        # Check if cuda is available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Define random seeds for reproducibility (if needed)
        random_seed(200, device)

        # Get the image data and age values
        image_label_generator = self.data_preprocessor.preprocess(
            self.data_loader.get_images(which='train'),
            self.data_loader.get_age_values(which='train'))

        # Convert the numpy arrays to tensors
        # image_data = torch.from_numpy(image_data)
        # age_value = torch.from_numpy(np.array(age_value))

        # print(image_data.shape, age_value.shape)

        # Load the model architecture
        #model = SFCN_mod(trained_weights=trained_weights, device=device, age_range=age_range)