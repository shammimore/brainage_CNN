"""Data model predictor."""

# %% External package import

from torch import cuda, device
from pathlib import Path

# %% Internal package import

from brainage.models.model_classes import (RankResnetModel, RankSFCNModel,
                                           SFCNModel)
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

    learning_rate : float
        ...

    number_of_epochs : int
        ...

    batch_size : int
        ...

    early_stopping_rounds : int
        ...

    reduce_lr_on_plateau : dict
        ...

    train_all_layers : bool
        ...

    architecture : string
        ...

    optimizer : string
        ...

    pretrained_weights : ...
        ...

    save_label : string
        ...

    Attributes
    ----------
    number_of_epochs : int
        See 'Parameters'.

    batch_size : int
        See 'Parameters'.

    data_generator : ...
        ...

    model : ...
        ...

    save_path : ...
        ...

    Methods
    -------
    - ``fit()`` : fit the prediction model;
    - ``tune_hyperparameters()`` : tune the model hyperparameters;
    - ``predict(image)`` : predict the brain age value(s) from the image(s).
    """

    def __init__(
            self,
            data_loader,
            data_preprocessor,
            learning_rate,
            number_of_epochs,
            batch_size,
            early_stopping_rounds,
            reduce_lr_on_plateau,
            train_all_layers,
            architecture,
            optimizer,
            pretrained_weights,
            save_label):

        print("\n\t Initializing the data model predictor ...")
        print("\t\t >>> Learning rate: {} - Number of epochs: {} - "
              "Batch size: {} - Early stopping rounds: {} - "
              "Reduce LR on plateau: {} - Train all layers: {} - "
              "Architecture: '{}' - Optimizer: '{}' - "
              "Pretrained weights: '{}' <<<"
              .format(learning_rate, number_of_epochs, batch_size,
                      early_stopping_rounds, reduce_lr_on_plateau,
                      train_all_layers, architecture, optimizer,
                      pretrained_weights))

        # Check if cuda is available
        comp_device = device(
            'cuda:0' if cuda.is_available() else 'cpu')

        # Define random seeds for reproducibility (if needed)
        random_seed(200, comp_device)

        # Get the attributes from the arguments
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        try:

            # Create a generator for the training data (image, age, fold)
            self.data_generator = data_preprocessor.preprocess(
                data_loader.get_images(which='train'),
                data_loader.get_age_values(which='train'),
                data_loader.get_fold_numbers(which='train'))

        except (AttributeError, TypeError):

            # Create an empty generator
            self.data_generator = iter(())

        # Initialize the prediction model
        architectures = {'rank_resnet3d': RankResnetModel,
                         'rank_sfcn': RankSFCNModel,
                         'sfcn': SFCNModel}
        self.model = architectures[architecture](pretrained_weights,
                                                 comp_device,
                                                 data_loader.age_filter)

        # Check if only the output layer should be fitted
        if not train_all_layers:

            # Set the gradient calculation to True for the output layer only
            self.model.freeze_inner_layers()

        # Check if the age filter is greater than the default range
        if (data_loader.age_filter != [42, 82]
                or architecture in ('rank_sfcn', 'rank_resnet3d')):

            # Get the age filter
            age_filter = data_loader.age_filter

            # Adapt the output layer to cover the modified age range
            self.model.adapt_output_layer(age_filter[1]-age_filter[0])

        # Set the optimizer for the model fitting
        self.model.set_optimizer(optimizer, learning_rate)

        # Send the model to the device (CPU or GPU)
        self.model.to(comp_device)

        # Create a directory to save the results in a folder
        self.save_path = Path('./brainage/models/exports/model_results',
                              save_label)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def fit(self):
        """Fit the prediction model."""
        # Check if the data generator is not available
        if not hasattr(self, 'data_generator'):

            raise AttributeError("Please specify the input data with the "
                                 "'data_path' argument, otherwise model "
                                 "fitting is not possible.")

        # Call the specific fit method of the model
        self.model.fit(self.data_generator, self.number_of_epochs,
                       self.batch_size, self.early_stopping_rounds,
                       self.reduce_lr_on_plateau, self.save_path)

    def tune_hyperparameters(self):
        """Tune the model hyperparameters."""
        pass

    def predict(
            self,
            image):
        """
        Predict the brain age value(s) from the image(s).

        Parameters
        ----------
        image : ...
            ...

        Returns
        -------
        ...
        """
        return self.model.forward(image)
