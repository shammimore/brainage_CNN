"""Brain age prediction."""

# %% External package import

from nibabel import load
from numpy import vstack

# %% Internal package import

from brainage.data import DataLoader
from brainage.preprocessing import DataPreprocessor
from brainage.models import DataModelPredictor
from brainage.evaluation import ModelEvaluator
from brainage.visualization import Visualizer
from brainage.tools import check_brain_age_predictor

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

    learning_rate : float, default=0.0001
        ...

    number_of_epochs : int, default=240
        ...

    batch_size : int, default=3
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
    decide whether to train all layers (True) or only the last layer (False).

    Examples
    --------
    First, the brain age predictor needs to be initialized. This can be done \
    via

    >>> bap = BrainAgePredictor(
            data_path=<your_data_path>,
            age_filter=[42, 82],
            image_dimensions=(160, 192, 160),
            steps=('normalize_image', 'crop_center'),
            learning_rate=0.0001,
            number_of_epochs=30,
            batch_size=3,
            train_all_layers=False,
            architecture='sfcn',
            optimizer='adam',
            pretrained_weights=<your_pretrained_weights_path>,
            metrics=('CORR', 'MSE', 'MAE'),
            save_label='trained_model')

    Model fitting is triggered by calling

    >>> bap.fit()

    A ready-to-use model can then predict the age based on input images, with \
    the syntax

    >>> prediction = bap.predict(
            image_path=<your_predict_image_path>)

    Evaluation of the model performance based on the passed ``metrics`` \
    argument is run from the lines

    >>> bap.evaluate(
            true_labels=<your_true_labels>,
            predicted_labels=prediction)

    Finally, different results can be plotted using

    >>> bap.plot("<name of the plot>")
    """

    def __init__(
            self,
            data_path=None,
            age_filter=[42, 82],
            image_dimensions=(160, 191, 160),
            steps=(),
            learning_rate=0.0001,
            number_of_epochs=240,
            batch_size=3,
            train_all_layers=False,
            architecture='sfcn',
            optimizer='adam',
            pretrained_weights=None,
            metrics=('CORR', 'MAE', 'MSE'),
            save_label='trained_model'):

        print('\n------ BRAIN AGE PREDICTOR ------\n')
        print('\t You are running the brain age predictor v0.1.0 ...')

        # Check inputs for initialization
        check_brain_age_predictor(**{key: value
                                     for key, value in locals().items()
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
            train_all_layers=train_all_layers,
            architecture=architecture,
            optimizer=optimizer,
            pretrained_weights=pretrained_weights,
            save_label=save_label)

        # Initialize the model evaluator
        self.model_evaluator = ModelEvaluator(metrics=metrics)

        # Initialize the visualizer
        self.visualizer = Visualizer(
            tracker=self.data_model_predictor.model.tracker,
            save_path=self.data_model_predictor.save_path)

    def fit(self):
        """Fit the prediction model."""
        self.data_model_predictor.fit()

    def predict(
            self,
            image_path):
        """Predict the brain age from an image."""
        print('\n\t Predicting the brain age ...')

        # Run single prediction
        if isinstance(image_path, str):

            image = load(image_path)
            preprocessed_image = self.data_preprocessor.run_pipeline(image)
            prediction = self.data_model_predictor.predict(
                preprocessed_image)

            return prediction

        # Run multiple predictions
        elif isinstance(image_path, (tuple, list)):

            images = (load(path) for path in image_path)
            preprocessed_images = (self.data_preprocessor.run_pipeline(image)
                                   for image in images)
            predictions = vstack([
                self.data_model_predictor.predict(image)
                for image in preprocessed_images])

            return predictions

    def evaluate(
            self,
            true_labels,
            predicted_labels):
        """Evaluate the model performance."""
        print('\n\t Evaluating the model performance ...')
        return self.model_evaluator.compute_metrics(true_labels,
                                                    predicted_labels)

    def plot(
            self,
            name):
        """Open the plot with the label 'name'."""
        print('\n\t Opening the plot "{}" ...'.format(name))
        self.visualizer.open_plot(name)
