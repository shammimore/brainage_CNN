"""Data preprocessing."""

# %% External package import

from numpy import array, expand_dims
import torch

# %% Internal package import

from brainage.tools import crop_center

# %% Class definition


class DataPreprocessor():
    """
    Data preprocessing class.

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
            image_dimensions):

        self.image_dimensions = image_dimensions
        self.steps = None

    def preprocess(
            self,
            images,
            age_values):
        """Preprocess the images."""

        def preprocess_single_image(image):
            """Preprocess a single image file."""
            # Load the subset images
            image_data = image.get_fdata()

            # Normalize the image data
            image_data = image_data / image_data.mean()

            # Crop the image
            image_data = crop_center(image_data, self.image_dimensions)

            # Add dimension to the subset images
            image_data = expand_dims(image_data, axis=0)

            return image_data

        image_label_generator = (
            (torch.from_numpy(preprocess_single_image(el[0])),
             torch.from_numpy(array(el[1]))) for el in zip(images, age_values))

        return image_label_generator
