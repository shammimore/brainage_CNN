"""Checking functions to check inputs."""

# %% External package import

from os.path import exists

# %% Function definitions


def check_brain_age_predictor(data_path,
            age_filter,
            image_dimensions,
            steps,
            learning_rate,
            number_of_epochs,
            batch_size,
            train_all_layers,
            architecture,
            optimizer,
            pretrained_weights,
            metrics,
            save_label):
    """
    Check input parameters for _brain_age_predictor class
    """

    # Check if neither data path nor weights are passed
    if not data_path and not pretrained_weights:
        raise ValueError("Please provide either a training dataset for "
                        "the model or pretrained model weights!")

    # Check if no weights are passed and train_all_layers is False
    if not pretrained_weights and not train_all_layers:
        raise ValueError("Please set 'train_all_layers' to True if no "
                        "pretrained weights are passed!")
    
    # Check age filter
    if not type(age_filter) == list or not len(age_filter) == 2 \
        or not age_filter[1] > age_filter[0]:
        raise ValueError("Please set 'age_filter' to list with "
                        "lower and upper bound for age range!")

    # Check for valid preprocessing steps
    steps_tuple = ('normalize_image', 'crop_center')
    for step in steps:
        if step not in steps_tuple:
            raise ValueError(f"{step} is not a valid preprocessing step. "
                    f"Please provide valid step from {steps_tuple}!")

    # Check learning_rate, number_of_epochs, batch_size
    if not isinstance(learning_rate, float) or not 0 < learning_rate < 1:
        raise ValueError("Please set 'learning_rate' between 0 and 1!")

    if not isinstance(number_of_epochs, int) or number_of_epochs <= 0:
        raise ValueError("Please set 'number_of_epochs' to a positive integer!")

    # Add check for batch size < total training sample size
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Please set 'batch_size' to a positive integer!")
    
    # Check train_all_layers
    if not isinstance(train_all_layers, bool):
        raise ValueError("Please set 'train_all_layers' to type bool!")


    # Check image_dimensions given the architecture
    if architecture == 'sfcn':
        if not image_dimensions == (160, 192, 160):
            raise ValueError("Please set 'image_dimensions' to (160, 192, 160) "
                             " to use SFCN architecture!")

    # Check for valid optimizers
    if not isinstance(optimizer, str):
        raise ValueError("Please set 'optimizer' to a string!")

    optimizers_tuple = ('sgd', 'adam')
    if optimizer not in optimizers_tuple:
        raise ValueError(f"{optimizer} is not a valid optimizer. "
                        "Please provide valid optimizer from "
                        f"{optimizers_tuple}")

    # Check pretrained weights is string and file exists
    if not isinstance(pretrained_weights, str):
        raise ValueError("Please set 'pretrained_weights' to a string!")
    
    if not exists(pretrained_weights):
     raise FileExistsError(f"{pretrained_weights} file does not exists "
                           "Please provide a valid pretrained_weights file")

    # Check performance metrics
    metrics_tuple = ('CORR', 'MSE', 'MAE')
    for metric in metrics:
        if metric not in metrics_tuple:
            raise ValueError(f"{metric} is not a valid metric. "
                             "Please provide valid metrics from "
                             f"{metrics_tuple}")

    # Check save_label






