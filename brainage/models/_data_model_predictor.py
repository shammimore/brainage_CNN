"""Data model predictor."""

# %% External package import

from pathlib import Path
import torch

# %% Internal package import

from brainage.models.model_classes import SFCNModel
from brainage.tools import random_seed

# %% Class definition


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

        # Convert the 
        pretrained_weights = Path(pretrained_weights)

        # Check if cuda is available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Define random seeds for reproducibility (if needed)
        random_seed(200, device)

        # Get the image data and age values
        image_label_generator = self.data_preprocessor.preprocess(
            self.data_loader.get_images(which='train'),
            self.data_loader.get_age_values(which='train'))

        # Initialize the prediction model
        architectures = {'sfcn': SFCNModel}
        model = architectures[self.architecture](pretrained_weights, device)

        if pretrained_weights.name == 'run_20190719_00_epoch_best_mae.p' or pretrained_weights.name == 'run_20190914_10_epoch_best_mae.p':
            model.architecture.load_state_dict(torch.load(pretrained_weights, map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.load(pretrained_weights, map_location=torch.device(device)))

        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
