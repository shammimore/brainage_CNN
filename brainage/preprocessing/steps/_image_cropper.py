"""Image cropper."""

# %% Internal package import

from brainage.tools import crop_center

# %% Class definition


class ImageCropper():
    """
    Image cropper class.

    This class provides ...

    Attributes
    ----------
    label : string
        ...

    Methods
    -------
    - ``transform(image_data, image_dimensions)`` : apply the transformation \
        (cropping) to the image.
    """

    label = 'crop_center'

    def transform(
            self,
            image_data,
            image_dimensions):
        """
        Apply the transformation (cropping) to the image.

        Parameters
        ----------
        image_data : ...
            ...

        image_dimensions : ...
            ...

        Returns
        -------
        ...
        """
        return crop_center(image_data, image_dimensions)
