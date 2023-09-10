"""Brain age prediction."""

# %% External package import

from nibabel import load as image_load
from numpy import vstack
from pathlib import Path
from time import gmtime, time, strftime
from torch import cuda, device, load

# %% Internal package import

from brainage.data import DataLoader
from brainage.preprocessing import DataPreprocessor
from brainage.models import DataModelPredictor
from brainage.evaluation import ModelEvaluator
from brainage.visualization import Visualizer
from brainage.tools import check_inputs

# %% Class definition


class BrainAgePredictor():
    """
    Brain age prediction class.

    This class provides ...

    Parameters
    ----------
    data_path : string, default=None
        ...

    age_filter : list, default=[42, 82]
        ...

    image_dimensions : tuple, default=(160, 192, 160)
        ...

    steps : tuple, default=()
        ...

    learning_rate : float, default=0.001
        ...

    number_of_epochs : int, default=100
        ...

    batch_size : int, default=4
        ...

    early_stopping_rounds : int, default=5
        ...

    reduce_lr_on_plateau : dict
        ...

    train_all_layers : bool, default=False
        ...

    architecture : string, default='sfcn'
        ...

    optimizer : string, default='adam'
        ...

    pretrained_weights : string, default=None
        ...

    metrics : tuple, default=('CORR', 'MAE', 'MSE')
        ...

    save_label : string, default='autosave'
        ...

    Attributes
    ----------
    data_loader : ...
        ...

    data_preprocessor : ...
        ...

    data_model_predictor : ...
        ...

    model_evaluator : ...
        ...

    visualizer : ...
        ...

    Methods
    -------
    - ``fit()`` : fit the prediction model;
    - ``predict(image_path)`` : predict the brain age from an image (or \
        multiple images in parallel);
    - ``evaluate(true_labels, predicted_labels)`` : evaluate the model \
        performance;
    - ``plot(name)`` : open the plot with the label 'name'.

    Notes
    -----
    You either need to specify the path to your data file (a .csv-file \
    containing image paths and age values) or pass on a file with the \
    pretrained network weights. If no data path is provided, input arguments \
    related to data preprocessing and model fitting have no effect, and \
    passing pretrained weights is mandatory. If a data path is provided, but \
    no pretrained weights are passed, then ``train_all_layers`` must be True. \
    If a data path is provided, and pretrained weights are passed, you can \
    decide whether to retrain all layers (True) or only the last layer (False).

    Examples
    --------
    First, the brain age predictor needs to be initialized. Using default \
    settings, this step is accomplished by

    >>> bap = BrainAgePredictor(
            data_path=<your_data_path>,
            age_filter=[42, 82],
            image_dimensions=(160, 192, 160),
            steps=('normalize_image', 'crop_center'),
            learning_rate=0.001,
            number_of_epochs=100,
            batch_size=4,
            early_stopping_rounds=10,
            reduce_lr_on_plateau={'rounds': 5, 'factor': 0.5},
            train_all_layers=False,
            architecture='sfcn',
            optimizer='adam',
            pretrained_weights=<your_pretrained_weights_path>,
            metrics=('CORR', 'MSE', 'MAE'),
            save_label='trained_model')

    Model fitting is triggered by calling

    >>> bap.fit()

    A ready-to-use model enables the prediction of age values given some \
    input images, using the syntax

    >>> prediction = bap.predict(image_path=<your_image_path>)

    Evaluation of the model performance based on the passed ``metrics`` \
    argument is performed by the line

    >>> bap.evaluate(true_labels=<your_true_labels>,
                     predicted_labels=prediction)

    Finally, different types of plots can be generated with

    >>> bap.plot(name=<plot_name>)
    """

    def __init__(
            self,
            data_path=None,
            age_filter=[42, 82],
            image_dimensions=(160, 192, 160),
            steps=(),
            learning_rate=0.001,
            number_of_epochs=100,
            batch_size=4,
            early_stopping_rounds=5,
            reduce_lr_on_plateau={'rounds': 3, 'factor': 0.5},
            train_all_layers=False,
            architecture='sfcn',
            optimizer='adam',
            pretrained_weights=None,
            metrics=('CORR', 'MAE', 'MSE'),
            save_label='autosave'):

        print('\n------ BRAIN AGE PREDICTOR ------\n')
        print('\t You are running the brain age predictor v0.1.0 ...')

        # Check inputs for initialization
        check_inputs(**{key: value for key, value in locals().items()
                        if key != 'self'})

        # Initialize the data loader
        self.data_loader = DataLoader(
            data_path=data_path,
            age_filter=age_filter)

        # Initialize the data preprocessor
        self.data_preprocessor = DataPreprocessor(
            image_dimensions=image_dimensions,
            steps=steps)

        # Initialize the data model predictor
        self.data_model_predictor = DataModelPredictor(
            data_loader=self.data_loader,
            data_preprocessor=self.data_preprocessor,
            learning_rate=learning_rate,
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            early_stopping_rounds=early_stopping_rounds,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            train_all_layers=train_all_layers,
            architecture=architecture,
            optimizer=optimizer,
            pretrained_weights=pretrained_weights,
            save_label=save_label)

        # Initialize the model evaluator
        self.model_evaluator = ModelEvaluator(
            metrics=metrics)

        # Initialize the visualizer
        self.visualizer = Visualizer(
            tracker=self.data_model_predictor.model.tracker,
            save_path=self.data_model_predictor.save_path)

    def fit(self):
        """Fit the prediction model."""
        if self.data_loader.sets['train'] is None:
            raise ValueError(
                "\t Model fitting is not possible - please provide some "
                "training data ...")

        # Start the runtime recording
        fit_start = time()

        # Fit the data model
        self.data_model_predictor.fit()

        # End the runtime recording and print the fitting time
        print("\n\t Fitting time is {} ...".format(
                strftime("%H:%M:%S", gmtime(time()-fit_start))))

    def predict(
            self,
            image_path):
        """
        Predict the brain age from an image.

        Parameters
        ----------
        image_path : string, tuple or list
            ...

        Returns
        -------
        ndarray
            ...
        """
        print('\n\t Predicting the brain age ...')

        # Check if the image path is a string
        if isinstance(image_path, str):

            # Load the image from the path
            image = image_load(image_path)

            # Preprocess the image
            preprocessed_image = self.data_preprocessor.run_pipeline(image)

            # Get the age value from the image
            prediction = self.data_model_predictor.predict(
                preprocessed_image)

            return prediction

        # Else, check if the image path is a tuple or a list
        elif isinstance(image_path, (tuple, list)):

            # Load all images from the paths
            images = (image_load(path) for path in image_path)

            # Preprocess all images
            preprocessed_images = (self.data_preprocessor.run_pipeline(image)
                                   for image in images)

            # Get the age values from all images and stack them
            predictions = vstack([
                self.data_model_predictor.predict(image)
                for image in preprocessed_images])

            return predictions.reshape(-1)

    def evaluate(
            self,
            true_labels,
            predicted_labels):
        """
        Evaluate the model performance.

        Parameters
        ----------
        true_labels : ...
            ...

        predicted_labels : ...
            ...

        Returns
        -------
        dict
            ...
        """
        print('\n\t Evaluating the model performance ...')

        return self.model_evaluator.compute_metrics(true_labels,
                                                    predicted_labels)

    def plot(
            self,
            name):
        """Open a specific plot type.

        Parameters
        ----------
        name : string
            ...
        """
        print('\n\t Opening "{}" plot ...'.format(name))

        # Open 'name' plot with the visualizer
        self.visualizer.open_plot(name)

    def update_parameters(
            self,
            trained_parameters):
        """Update the parameters of the model."""
        # Get the device
        comp_device = device('cuda:0' if cuda.is_available() else 'cpu')
        print(comp_device)
        print(trained_parameters)

        # Load the parameters
        self.data_model_predictor.model.architecture.load_state_dict(load(Path(trained_parameters), map_location=device(comp_device)))

        # Send the model to the device
        self.data_model_predictor.model.to(comp_device)
