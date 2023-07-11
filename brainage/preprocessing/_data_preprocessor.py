"""Data preprocessing."""

# %% External package import

from numpy import expand_dims
from torch import from_numpy

# %% Internal package import

from brainage.preprocessing.steps import ImageCropper, ImageNormalizer

# %% Class definition


class DataPreprocessor():
    """
    Data preprocessing class.

    This class provides ...

    Parameters
    ----------
    image_dimensions : ...
        ...

    steps : tuple
        ...

    Attributes
    ----------
    image_dimensions : ...
        ...

    steps : tuple
        See `Parameters`.

    steps_catalogue : tuple
        ...

    pipeline : ...
        ...

    Methods
    -------
    - ``build()`` : ...
    - ``run_pipeline(image)`` : ...
    - ``preprocess(images, age_values)`` : ...
    """

    def __init__(
            self,
            image_dimensions,
            steps):

        # Get the attributes from the argument
        self.image_dimensions = image_dimensions
        self.steps = steps

        # Specify the preprocessing steps catalogue
        self.steps_catalogue = (ImageCropper, ImageNormalizer)

        # Build the preprocessing pipeline
        self.pipeline = self.build()

    def build(self):
        """
        Build the preprocessing pipeline.

        Returns
        -------
        tuple
            ...
        """
        return tuple(step_class()
                     for step_class in self.steps_catalogue
                     for step in self.steps
                     if step_class.label == step)

    def run_pipeline(
            self,
            image):
        """
        Run the preprocessing pipeline.

        Parameters
        ----------
        image : ...
            ...

        Returns
        -------
        image_data : ...
            ...
        """
        # Extract the image data
        image_data = image.get_fdata()

        # Run the preprocessing steps sequentially
        for step in self.pipeline:
            image_data = step.transform(image_data, self.image_dimensions)

        # Add dimension to the image data
        image_data = expand_dims(image_data, axis=0)

        return from_numpy(image_data)

    def preprocess(
            self,
            images,
            age_values,
            folds):
        """
        Preprocess the images.

        Parameters
        ----------
        images : ...
            ...

        age_values : ...
            ...

        Returns
        -------
        image_label_generator : ...
            ...
        """
        def preprocess_single_image(image):
            """
            Preprocess a single image file.

            Parameters
            ----------
            image : ...
                ...

            Returns
            -------
            image_data : ...
                ...
            """
            # Run the preprocessing pipeline
            image_data = self.run_pipeline(image)

            return image_data

        # Build the image-label pair generator
        image_label_generator = (
            (preprocess_single_image(el[0]), el[1], el[2])
            for el in zip(images, age_values, folds))

        return image_label_generator
