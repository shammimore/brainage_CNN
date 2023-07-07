"""Data model predictor."""

# %% External package import

from pathlib import Path
from torch import cuda, device

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
    data_loader : ...
        ...

    data_preprocessor : ...
        ...

    learning_rate : ...
        ...

    number_of_epochs : ...
        ...

    batch_size : ...
        ...

    train_all_layers : ...
        ...

    architecture : ...
        ...

    optimizer : ...
        ...

    pretrained_weights : ...
        ...

    Attributes
    ----------
    model : ...
        ...

    Methods
    -------
    - ``fit(data, number_of_epochs, batch_size)`` : ...
    - ``tune_hyperparameters()`` : ...
    - ``predict(image)`` : ...
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

        # Check if cuda is available
        comp_device = device(
            'cuda:0' if cuda.is_available() else 'cpu')

        # Define random seeds for reproducibility (if needed)
        random_seed(200, comp_device)

        # Get the image data and age values
        image_label_generator = data_preprocessor.preprocess(
            data_loader.get_images(which='train'),
            data_loader.get_age_values(which='train'))

        # Create a path object from the link to the pretrained weights
        pretrained_weights = Path(pretrained_weights)

        # Initialize the prediction model
        architectures = {'sfcn': SFCNModel}
        self.model = architectures[architecture](pretrained_weights,
                                                 comp_device,
                                                 data_loader.age_filter)

        # Check if only the output layer should be fitted
        if not train_all_layers:

            # Set the gradient calculation to True for the output layer only
            self.model.freeze_inner_layers()

        # Check if the age filter is greater than the default range
        if data_loader.age_filter != [42, 82]:

            # Get the age filter
            age_filter = data_loader.age_filter

            # Adapt the output layer to cover the new age range
            self.model.adapt_output_layer(age_filter[1]-age_filter[0])

        # Set the optimizer for the model fitting
        self.model.set_optimizer(optimizer, learning_rate)

        # Send the model to the device (CPU or GPU)
        self.model.to(comp_device)

        # Fit the model
        self.fit(image_label_generator, number_of_epochs, batch_size)

    def fit(
            self,
            data,
            number_of_epochs,
            batch_size):
        """Fit the prediction model."""
        self.model.fit(data, number_of_epochs, batch_size)

    def tune_hyperparameters(self):
        """Tune the model hyperparameters."""
        pass

    def predict(
            self,
            image):
        """Generate an age prediction on a single image."""
        return self.model.forward(image)
