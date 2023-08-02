"""Data preprocessing."""

# %% External package import

from numpy import expand_dims

# %% Internal package import

from brainage.preprocessing.steps import ImageCropper, ImageNormalizer

# %% Class definition


class DataPreprocessor():
    """
    Data preprocessing class.

    This class provides ...

    Parameters
    ----------
    image_dimensions : tuple
        ...

    steps : tuple
        ...

    Attributes
    ----------
    image_dimensions : tuple
        ...

    steps : tuple
        See `Parameters`.

    pipeline : tuple
        ...

    Methods
    -------
    - ``build()`` : build the preprocessing pipeline;
    - ``run_pipeline(image)`` : run the preprocessing pipeline;
    - ``preprocess(images, age_values)`` : preprocess the images.
    """

    def __init__(
            self,
            image_dimensions,
            steps):

        print('\n\t Initializing the data preprocessor ...')
        print('\t\t >>> Image dimensions: {} - Steps: {} <<<'.format(
            image_dimensions, steps))

        # Get the attributes from the arguments
        self.image_dimensions = image_dimensions
        self.steps = steps

        # Build the preprocessing pipeline
        self.pipeline = self.build((ImageCropper, ImageNormalizer))

    def build(
            self,
            steps_catalogue):
        """
        Build the preprocessing pipeline.

        Parameters
        ----------
        steps_catalogue : tuple
            ...

        Returns
        -------
        tuple
            ...
        """
        print('\t\t Building the preprocessing pipeline ...')

        return tuple(step_class()
                     for step in self.steps
                     for step_class in steps_catalogue
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
        image_data : ndarray
            ...
        """
        # Extract the image data
        image_data = image.get_fdata()

        # Run the preprocessing steps sequentially
        for step in self.pipeline:
            image_data = step.transform(image_data, self.image_dimensions)

        # Add a dimension to the image data
        image_data = expand_dims(image_data, axis=0)

        return image_data

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

        folds : ...
            ...

        Returns
        -------
        image_label_generator : ...
            ...
        """
        print('\t\t Preprocessing the images ...')

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

        # Build the image-label-fold triplet generator
        image_label_generator = (
            (preprocess_single_image(el[0]), el[1], el[2])
            for el in zip(images, age_values, folds))

        return image_label_generator
