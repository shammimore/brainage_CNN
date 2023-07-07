"""Brain age prediction."""

# %% External package import

from nibabel import load
from numpy import hstack

# %% Internal package import

from brainage.data import DataLoader
from brainage.preprocessing import DataPreprocessor
from brainage.models import DataModelPredictor
from brainage.evaluation import ModelEvaluator

# %% Class definition


class BrainAgePredictor():
    """
    Brain age prediction class.

    This class provides ...

    Parameters
    ----------
    data_path : ...
        ...

    age_filter : ...
        ...

    image_dimensions : ...
        ...

    steps : ...
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
    data_loader : ...
        ...

    data_preprocessor : ...
        ...

    data_model_predictor : ...
        ...

    Methods
    -------
    - ``predict(image_path, age)`` : ...
    """

    def __init__(
            self,
            data_path,
            age_filter,
            image_dimensions,
            steps=(),
            learning_rate=0.0001,
            number_of_epochs=240,
            batch_size=3,
            train_all_layers=False,
            architecture='sfcn',
            optimizer='adam',
            pretrained_weights=None,
            metrics=()):

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
            pretrained_weights=pretrained_weights)

        # Initialize the model evaluator
        self.model_evaluator = ModelEvaluator(metrics)

    def predict(
            self,
            image_path):
        """Predict the brain age from an image."""
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
            predictions = hstack([
                self.data_model_predictor.predict(image)
                for image in preprocessed_images])

            return predictions

    def evaluate(
            self,
            true_labels,
            predicted_labels):
        """Evaluate the model performance."""
        return self.model_evaluator.compute_metrics(true_labels,
                                                    predicted_labels)
