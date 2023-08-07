"""Additional functions for class support."""

# %% External package import

from numpy import arange, array, floor, isscalar, ndim, random, vstack, zeros
from random import seed
from scipy.stats import norm
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed as cuda_manual_seed
from torch.cuda import manual_seed_all
from torch.backends import cudnn

# %% Function definitions


def convert_number_to_vector(age_values, bin_range, bin_step, sigma):
    """
    Convert numbers (age values) into vectors ("soft" labels).

    Parameters
    ----------
    age_values : ...
        ...

    bin_range : tuple
        ...

    bin_step : ...
        ...

    sigma : ...
        ...

    Returns
    -------
    ...
        ...

    Notes
    -----
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    # Get start and end from the bin range
    bin_start = bin_range[0]
    bin_end = bin_range[1]

    # Get the bin length
    bin_length = bin_end - bin_start

    # Check if the bin length is not a multiple of the bin step
    if not bin_length % bin_step == 0:

        # Print a statement on the non-divisibility and return -1
        print("Bin's range should be divisible by bin_step!")
        return -1

    # Compute the number of bins
    bin_number = int(bin_length / bin_step)

    # Get the bin centers
    bin_centers = (bin_start + float(bin_step) / 2
                   + bin_step * arange(bin_number))

    # Check if sigma is zero
    if sigma == 0:

        # Convert the age value into an array
        age_values = array(age_values)

        # Get the index as "soft" label
        i = floor((age_values - bin_start) / bin_step)
        i = i.astype(int)

        return i, bin_centers

    # Else, check if sigma is greater than zero
    elif sigma > 0:

        # Check if the age value is a scalar
        if isscalar(age_values):

            # Initialize the "soft" labels
            v = zeros((bin_number,))

            # Loop over the number of bins
            for i in range(bin_number):

                # Get the lower and upper bound of the bin
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2

                # Compute the Gaussian cdf for the bounds
                cdfs = norm.cdf([x1, x2], loc=age_values, scale=sigma)

                # Get the "soft" label from the cdf difference
                v[i] = cdfs[1] - cdfs[0]

            return v, bin_centers

        else:

            # Initialize the "soft" labels for multiple age values
            v = zeros((len(age_values), bin_number))

            # Loop over the number of age values and bins
            for j in range(len(age_values)):
                for i in range(bin_number):

                    # Get the lower and upper bound of the bin
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2

                    # Compute the Gaussian cdf for the bounds
                    cdfs = norm.cdf([x1, x2], loc=age_values[j], scale=sigma)

                    # Get the "soft" label from the cdf difference
                    v[j, i] = cdfs[1] - cdfs[0]

            return v, bin_centers


def crop_center(data, output_shape):
    """
    Return the center part of volume data.

    Parameters
    ----------
    data : ...
        ...

    output_shape : ...
        ...

    Returns
    -------
    data_crop : ...
        ...
    """
    # Get the data shape
    shape = data.shape

    # Get the number of data dimensions
    number_of_dimensions = ndim(data)

    # Get the cropping area per image axis
    x_crop = int((shape[-1] - output_shape[-1]) / 2)
    y_crop = int((shape[-2] - output_shape[-2]) / 2)
    z_crop = int((shape[-3] - output_shape[-3]) / 2)

    # Check if the number of dimensions is three
    if number_of_dimensions == 3:

        # Get the cropped 3D data
        return data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]

    # Else, check if the number of dimensions is four
    elif number_of_dimensions == 4:

        # Get the cropped 4D data
        return data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]

    # Else, check if the number of dimensions is five
    elif number_of_dimensions == 5:

        # Get the cropped 5D data
        return data[:, :, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]

    # Raise an error to indicate a wrong number of dimensions
    raise ('Wrong dimension! dim=%d.' % number_of_dimensions)


def extend_label_to_vector(age_values, bin_range):
    """
    Convert the labels into extended binary vectors.

    Parameters
    ----------
    age_values : ...
        ...

    bin_range : ...
        ...

    Returns
    -------
    ...
        ...
    """
    return vstack([[true_age > bin_age
                   for bin_age in range(bin_range[0], bin_range[1]-1)]
                   for true_age in age_values])


def get_batch(generator, batch_size):
    """
    Get a single batch from a generator.

    Parameters
    ----------
    generator : ...
        ...

    batch_size : int
        ...

    Returns
    -------
    tuple
        ...
    """
    # Initialize the batch as a list
    batch = []

    # Loop over the batch size
    for i in range(batch_size):

        # Try to get and append the next element from the generator
        try:
            element = next(generator)
            batch.append(element)

        # Return the batch if the generator is used up
        except StopIteration:

            return batch, False

    return batch, True


def get_bin_centers(bin_range, bin_step):
    """
    Get the bin centers for prediction.

    Parameters
    ----------
    bin_range : ...
        ...

    bin_step : ...
        ...

    Returns
    -------
    ...
        ...
    """
    # Get the length of the bin
    bin_length = bin_range[1] - bin_range[0]

    # Get the number of bins
    bin_number = int(bin_length / bin_step)

    return bin_range[0] + float(bin_step) / 2 + bin_step * arange(bin_number)


def random_seed(seed_value, device):
    """
    Set the random seeds.

    Parameters
    ----------
    seed_value : ...
        ...

    device : ...
        ...
    """
    # Set the random seeds for the CPU variables
    random.seed(seed_value)
    torch_manual_seed(seed_value)

    # Set the random seed for Python
    seed(seed_value)

    # Set the random seeds for CUDA
    if device == "cuda:0":
        cuda_manual_seed(seed_value)
        manual_seed_all(seed_value)
        cudnn.deterministic = True
        cudnn.benchmark = False
