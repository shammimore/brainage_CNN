"""SFCN model."""

# %% External package import

from itertools import tee
from numpy import exp, expand_dims, Inf, vstack
from pathlib import Path
from torch import as_tensor, device, float32, load, no_grad, save
from torch.nn import Conv3d, DataParallel, init, Module
from torch.optim import Adam, SGD

# %% Internal package import

from brainage.models.architectures import SFCN
from brainage.models.loss_functions import KLDivLoss
from brainage.tools import convert_number_to_vector, get_batch, get_bin_centers

# %% Class definition


class SFCNModel(Module):
    """
    SFCN model class.

    This class provides ...

    Parameters
    ----------
    pretrained_weights : ...
        ...

    comp_device : ...
        ...

    age_filter : ...
        ...

    Attributes
    ----------
    comp_device : ...
        See 'Parameters'.

    age_filter : list
        See 'Parameters'.

    architecture : ...
        ...

    tracker : dict
        ...

    parameters : ...
        ...

    Methods
    -------
    - ``freeze_inner_layers()`` : freeze the parameters of the input and \
        hidden layers;
    - ``adapt_output_layer(age_range)`` : adapt the output layer for the age \
        range;
    - ``set_optimizer(optimizer, learning_rate)`` : set the optimizer for the \
        model fitting;
    - ``fit(data, number_of_epochs, batch_size, early_stopping_rounds, reduce_lr_on_plateau)`` : fit the SFCN model;
    - ``forward(image)`` : perform a single forward pass through the model.
    """

    def __init__(
            self,
            pretrained_weights,
            comp_device,
            age_filter):

        # Call the superclass constructor
        super(SFCNModel, self).__init__()

        # Get the attributes from the arguments
        self.comp_device = comp_device
        self.age_filter = age_filter

        # Initialize and parallelize the architecture
        self.architecture = DataParallel(SFCN())

        # Load the pretrained weights if applicable
        if pretrained_weights:
            self.architecture.load_state_dict(load(
                Path(pretrained_weights), map_location=device(comp_device)))

        # Initialize the tracking dictionary
        self.tracker = {}

    def freeze_inner_layers(self):
        """Freeze the parameters of the input and hidden layers."""
        print('\t\t Freezing the parameters of input and hidden layers ...')

        # Set the gradient calculation for all parameters to False
        for param in self.parameters():
            param.requires_grad = False

        # Set the gradient calculation for the output layer to True
        for param in self.architecture.module.classifier.conv_6.parameters():
            param.requires_grad = True

    def adapt_output_layer(
            self,
            age_range):
        """
        Adapt the output layer for the age range.

        Parameters
        ----------
        age_range : ...
            ...
        """
        print('\t\t Adapting the output layer for the age range ...')

        # Resize the output layer for the age range
        self.architecture.module.classifier.conv_6 = Conv3d(64,
                                                            age_range,
                                                            padding=0,
                                                            kernel_size=1)

        # Initialize the output layer's weights with He
        init.kaiming_normal_(self.architecture.module.classifier.conv_6.weight)

    def set_optimizer(
            self,
            optimizer,
            learning_rate):
        """
        Set the optimizer for the model fitting.

        Parameters
        ----------
        optimizer : string
            ...

        learning_rate : float
            ...
        """
        print('\t\t Setting the optimizer ...')

        # Check if the optimizer is 'adam'
        if optimizer == 'adam':

            # Initialize the Adam optimizer
            self.optimizer = Adam(filter(lambda p: p.requires_grad,
                                         self.parameters()),
                                  lr=learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0,
                                  amsgrad=False)

        # Else, check if the optimizer is 'sgd'
        elif optimizer == 'sgd':

            # Initialize the SGD optimizer
            self.optimizer = SGD(filter(lambda p: p.requires_grad,
                                        self.parameters()),
                                 lr=learning_rate,
                                 momentum=0.9,
                                 weight_decay=0.001)

    def fit(
            self,
            data,
            number_of_epochs,
            batch_size,
            early_stopping_rounds,
            reduce_lr_on_plateau,
            save_path):
        """
        Fit the SFCN model.

        Parameters
        ----------
        data : ...
            ...

        number_of_epochs : int
            ...

        batch_size : int
            ...

        early_stopping_rounds : int
            ...

        reduce_lr_on_plateau : dict
            ...

        save_path : ...
            ...
        """
        print('\n\t Fitting the SFCN model to the data ...')

        def get_input(batch):
            """Get the images, labels, soft labels and centers from a batch."""
            # Extract the images and labels from the batch
            images = vstack([sample[0] for sample in batch])
            labels = [sample[1] for sample in batch]

            # Transform the labels to soft labels (probabilities) with centers
            soft_labels, centers = convert_number_to_vector(
                age_values=labels, bin_range=self.age_filter,
                bin_step=1, sigma=1)

            # Convert the soft labels to tensors
            soft_labels = as_tensor(soft_labels, dtype=float32,
                                    device=self.comp_device)

            # Add a dimension to the images
            images = expand_dims(images, axis=1)

            # Convert the images to tensors
            images = as_tensor(images, dtype=float32, device=self.comp_device)

            return images, labels, soft_labels, centers

        def get_output(centers, model_output):
            """Get the age prediction from the model output."""
            # Shift the model output to CPU and convert into an array
            model_output = model_output.detach().cpu().numpy()

            # Take the exponential to convert into a probability (sigmoid)
            probability = exp(model_output)

            return probability @ centers

        def train(image, soft_labels):
            """Perform a single training step."""
            # Set the architecture into training mode
            self.architecture.train()

            # Clear all gradients
            self.optimizer.zero_grad()

            # Get the model output for the image
            model_output = self.architecture(image)

            # Reshape the model output
            model_output = model_output[0].reshape([image.shape[0], -1])

            # Compute the Kullback-Leibler divergence loss
            training_loss = KLDivLoss(model_output, soft_labels)

            # Propagate the loss back to the parameters
            training_loss.backward()

            # Perform a single parameter update
            self.optimizer.step()

            return training_loss, model_output

        def validate(image, soft_labels):
            """Perform a single validation step."""
            # Set the architecture into evaluation mode
            self.architecture.eval()

            # Get the model output with gradient calculation disabled
            with no_grad():
                model_output = self.architecture(image)

            # Reshape the model output
            model_output = model_output[0].reshape([image.shape[0], -1])

            # Compute the Kullback-Leibler divergence loss
            validation_loss = KLDivLoss(model_output, soft_labels)

            return validation_loss, model_output

        # Create training and validation generators for all epochs
        data_generators = tee(data, number_of_epochs*2)

        # Initialize the lists for the training/validation loss per epoch
        train_loss_per_epoch, val_loss_per_epoch = [], []

        # Initialize the minimum validation loss
        min_val_loss = Inf

        # Loop over the number of epochs
        for epoch in range(number_of_epochs):

            print('\n\t ------ Epoch %d ------\n' % (epoch+1))

            # Initialize the training and validation loss to zero
            train_loss_over_batches, val_loss_over_batches = 0, 0

            # Get training and validation data by filtering the fold number
            training_data = (el for el in data_generators[epoch*2]
                             if el[2] != 1)
            validation_data = (el for el in data_generators[epoch*2+1]
                               if el[2] == 1)

            # Set the continuation flag
            proceed = True

            # Set the counter
            counter = 0

            # Loop while the continuation flag is set to True
            while proceed:

                # Increment the counter
                counter += 1

                # Get a new batch from the training data generator
                batch, proceed = get_batch(training_data, batch_size)

                # Check if the batch is non-empty
                if len(batch) > 0:

                    # Get the model input
                    images, labels, soft_labels, centers = get_input(batch)

                    # Perform a single training step
                    training_loss, model_output = train(images, soft_labels)

                    # Add the training loss to the total batch loss
                    train_loss_over_batches += training_loss.item()

                    # Get the training prediction
                    training_prediction = get_output(centers, model_output)

                    # Print a message after the training epoch
                    print('\t Training - Batch: {} - Loss: {} - '
                          'Prediction: {} - Ground Truth: {}'
                          .format(counter, training_loss, training_prediction,
                                  labels))

            # Append the training loss for the epoch
            train_loss_per_epoch.append(train_loss_over_batches/counter)

            # Reset the continuation flag
            proceed = True

            # Reset the counter
            counter = 0

            # Loop while the continuation flag is set to True
            while proceed:

                # Increment the counter
                counter += 1

                # Get a new batch from the validation data generator
                batch, proceed = get_batch(validation_data, batch_size)

                # Check if the batch is non-empty
                if len(batch) > 0:

                    # Get the model input
                    image, labels, soft_labels, centers = get_input(batch)

                    # Perform a single validation step
                    validation_loss, model_output = validate(image,
                                                             soft_labels)

                    # Add the validation loss to the total batch loss
                    val_loss_over_batches += validation_loss.item()

                    # Get the validation prediction
                    validation_prediction = get_output(centers, model_output)

                    # Print a message after the validation epoch
                    print('\t Validation - Batch: {} - Loss: {} - '
                          'Prediction: {} - Ground Truth: {}'
                          .format(counter, validation_loss,
                                  validation_prediction, labels))

            # Append the validation loss for the epoch
            val_loss = val_loss_over_batches/counter
            val_loss_per_epoch.append(val_loss)

            # Check if the validation loss is reduced
            if val_loss < min_val_loss:
                print('\t Saving model - current loss: {}, previous \
                      minimum loss: {}'
                      .format(val_loss, min_val_loss))

                # Save the model state dictionary
                save(self.architecture.state_dict(),
                     Path(save_path, 'state_dict.pt'))

                # Update the current minimum validation loss
                min_val_loss = val_loss

                # (Re-)Set the counters for early stopping and LR reduction
                early_stopping_counter = 0
                reduce_lr_counter = 0

            # Else, check for early stopping or LR reduction
            else:

                # Increment the counter for early stopping
                early_stopping_counter += 1

                # Check if the number of early stopping rounds is reached
                if early_stopping_counter == early_stopping_rounds:

                    print("\t Terminating the model fitting due to early "
                          "stopping ...")

                    # Break the epoch loop
                    break

                # Increment the counter for LR reduction
                reduce_lr_counter += 1

                # Check if the number of LR reduction rounds is reached
                if reduce_lr_counter == reduce_lr_on_plateau['rounds']:

                    # Get the current learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']

                    print("\t Reducing the learning rate to {}"
                          .format(current_lr*reduce_lr_on_plateau['factor']))

                    # Overwrite the learning rate for each parameter group
                    for group in self.optimizer.param_groups:
                        group['lr'] = current_lr*reduce_lr_on_plateau['factor']

                    # Reset the counter for LR reduction
                    reduce_lr_counter = 0

        # Update the tracker with all variables of interest
        self.tracker.update({
            'epochs': epoch,
            'training_loss': train_loss_per_epoch,
            'validation_loss': val_loss_per_epoch,
            'model_state_dict': self.architecture.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()})

        # Save the tracker to a file
        save(self.tracker, Path(save_path, 'tracker.pt'))

    def forward(
            self,
            image):
        """
        Perform a single forward pass through the model.

        Parameters
        ----------
        image : ...
            ...

        Returns
        -------
        prediction : ...
            ...
        """
        # Get the image shape
        shape = image.shape

        # Reshape the image to 5D (with batch size 1)
        image = image.reshape(1, shape[0], shape[1], shape[2], shape[3])

        # Convert the image to a tensor
        image = as_tensor(image, dtype=float32, device=self.comp_device)

        # Set the architecture into evaluation mode
        self.architecture.eval()

        # Get the model output with gradient calculation disabled
        with no_grad():
            model_output = self.architecture(image)

        # Shift the output back to the CPU
        model_output = model_output[0].cpu().numpy().reshape([1, -1])

        # Get the bin centers
        centers = get_bin_centers(self.age_filter, 1)

        # Take the exponential to convert into a probability
        probability = exp(model_output)

        return probability @ centers
