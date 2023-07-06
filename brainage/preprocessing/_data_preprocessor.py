"""Data preprocessing."""

# %% External package import

from numpy import array, expand_dims
import torch

# %% Internal package import
from brainage.preprocessing.steps import ImageCropper, ImageNormalizer

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
            image_dimensions, steps=None):

        self.image_dimensions = image_dimensions

        steps_catalogue = (ImageCropper, ImageNormalizer)

        if steps is not None:
            self.steps = steps
        else:
            self.steps = ()
        
        self.pipeline = [step_class() for step_class in steps_catalogue if step_class.label == step for step in self.steps]
        print(self.pipeline)

    def preprocess(
            self,
            images,
            age_values):
        """Preprocess the images."""

        def preprocess_single_image(image):
            """Preprocess a single image file."""
            
            # Load the subset images
            image_data = image.get_fdata()

            # Apply user-defined preprocessing steps
            for preprocess_index, preprocess_step in  enumerate(self.pipeline): 
                print(preprocess_step) 
                image_data = self.pipeline[preprocess_index].transform(image_data, self.image_dimensions)

            # Add dimension to the subset images
            image_data = expand_dims(image_data, axis=0)

            return image_data

        image_label_generator = (
            (torch.from_numpy(preprocess_single_image(el[0])),
             torch.from_numpy(array(el[1]))) for el in zip(images, age_values))

        return image_label_generator

# %%
