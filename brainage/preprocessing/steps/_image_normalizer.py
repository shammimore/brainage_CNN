"""Image normalizer."""

# %% Class definition


class ImageNormalizer():
    """
    Image normalizer class.

    This class provides ...

    Attributes
    ----------
    label : string
        ...

    Methods
    -------
    - ``transform(image_data, _)`` : apply the transformation (normalization) \
        to the image.
    """

    label = 'normalize_image'

    def transform(
            self,
            image_data,
            _):
        """
        Apply the transformation (normalization) to the image.

        Parameters
        ----------
        image_data : ...
            ...

        _ : ...
            ...

        Returns
        -------
        ...
        """
        return image_data / image_data.mean()
