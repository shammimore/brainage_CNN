"""RankSFCN model."""

# %% External package import

from itertools import tee
from numpy import array, dot, expand_dims, vstack, Inf
from pathlib import Path
from torch import as_tensor, device, float32, load, no_grad, save
from torch import zeros
from torch.nn import DataParallel, Linear, Module, Parameter
from torch.optim import Adam, SGD

# %% Internal package import

from brainage.models.architectures import rankresnet34
from brainage.models.loss_functions import BCELoss
from brainage.tools import extend_label_to_vector, get_batch

# %% Class definition


class RankResnetModel(Module):
    """
    Rank Resnet model class.

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
        See `Parameters`.

    age_filter : list
        See `Parameters`.

    architecture : ...
        ...

    parameters : ...
        ...

    tracker : dict
        ...

    Methods
    -------
    - ``freeze_inner_layers()`` : freeze the parameters of the input and \
        hidden layers;
    - ``adapt_output_layer(age_range)`` : adapt the output layer for the age \
        range;
    - ``set_optimizer(optimizer, learning_rate)`` : set the optimizer for the \
        model training;
    - ``fit(data, number_of_epochs, batch_size)`` : fit the SFCN model;
    - ``forward(image)`` : perform a single forward pass through the model.
    """

    def __init__(
            self,
            pretrained_weights,
            comp_device,
            age_filter):

        # Call the superclass constructor
        super(RankResnetModel, self).__init__()

        # Get the attributes from the arguments
        self.comp_device = comp_device
        self.age_filter = age_filter

        # Initialize the model architecture
        self.architecture = rankresnet34()

        # Parallelize the model
        self.architecture = DataParallel(self.architecture)

        # Load the pretrained weights if applicable
        if pretrained_weights:
            self.architecture.load_state_dict(load(
                Path(pretrained_weights), map_location=device(comp_device)))

        # Initialize the tracking dictionary
        self.tracker = {}

    def freeze_inner_layers(self):
        """Freeze the parameters of the input and hidden layers."""
        print('\t\t Freezing the parameters of input and hidden layers ...')
        
        # Get all and only the output layer parameters
        all_params = self.parameters()
        out_params = self.architecture.module.classifier.parameters()

        # Set the gradient calculation for all parameters to False
        for param in all_params:
            param.requires_grad = False

        # Set the gradient calculation for the output layer to True
        for param in out_params:
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

        # Change the final linear layer without biases to the architecture
        self.architecture.module.classifier = Linear(512, 1,
                                                       bias=False)

        # Add the (trainable) linear bias vector to the architecture
        self.architecture.module.linear_bias = Parameter(
            zeros(age_range-1).float())

    def set_optimizer(
            self,
            optimizer,
            learning_rate):
        """
        Set the optimizer for the model training.

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
        """
        print('')
        print('\t Fitting the Rank-Consistent ResNet34 model to the data ...')

        def get_input(batch):
            """Get the images, soft labels and centers from a batch."""
            # Extract the images and labels from the batch
            images = vstack([sample[0] for sample in batch])
            labels = [sample[1] for sample in batch]

            # Transform the labels to extended binary vectors
            extended_labels = extend_label_to_vector(
                    x=labels, bin_range=self.age_filter)

            # Convert the soft labels to tensors
            extended_labels = as_tensor(extended_labels, dtype=float32,
                                        device=self.comp_device)

            # Add a dimension to the images
            images = expand_dims(images, axis=1)

            # Convert the images to tensors
            images = as_tensor(images, dtype=float32, device=self.comp_device)

            return images, labels, extended_labels

        def get_output(model_output):
            """Get the age prediction from the model output."""
            return dot(model_output.detach().cpu().numpy(), 
                       array(range(self.age_filter[0]+1, self.age_filter[1])))

        def train(image, extended_labels):

            # Set the architecture into training mode
            self.architecture.train()

            # Clear all gradients
            self.optimizer.zero_grad()

            # Get the model output for the image
            model_output = self.architecture(image)

            # Reshape the model output
            model_output = model_output[0].reshape([image.shape[0], -1])

            # Calculate the BCE loss
            training_loss = BCELoss(model_output, extended_labels)

            # Propagate the loss back to the parameters
            training_loss.backward()

            # Perform a single parameter update
            self.optimizer.step()

            # Clamp the weights of the output layer for non-negativity
            self.architecture.module.classifier.weight.data = (
                self.architecture.module.classifier.weight.data.clamp(min=0))

            return training_loss, model_output

        def validate(image, extended_labels):

            # Set the architecture into evaluation mode
            self.architecture.eval()

            # Get the model output with gradient calculation disabled
            with no_grad():
                model_output = self.architecture(image)

            # Reshape the model output
            model_output = model_output[0].reshape([image.shape[0], -1])

            # Calculate the BCE loss
            validation_loss = BCELoss(model_output, extended_labels)

            return validation_loss, model_output

        # Create training and validation generators for all epochs
        data_generators = tee(data, number_of_epochs*2)

        # Loop over the number of epochs
        train_loss_per_epoch, val_loss_per_epoch = [], []
        min_val_loss = Inf

        for epoch in range(number_of_epochs):

            # Initialize the train and validation loss to zero
            train_loss_over_batchs, val_loss_over_batchs = 0, 0

            print('\n\t ------ Epoch %d ------\n' % (epoch))

            # Get training and validation data by filtering the fold number
            training_data = (el for el in data_generators[epoch*2]
                             if el[2] != 1)
            validation_data = (el for el in data_generators[epoch*2+1]
                               if el[2] == 1)

            # Iterate over the training data batch-wise
            proceed = True  # Continuation flag
            counter = 1  # Batch counter

            while proceed:

                # Get a new batch from the training data generator
                batch, proceed = get_batch(training_data, batch_size)

                # Check if the batch is non-empty
                if len(batch) > 0:

                    # Get the model input
                    images, labels, extended_labels = get_input(batch)

                    # Perform a single training step
                    training_loss, model_output = train(images,
                                                        extended_labels)
                    train_loss_over_batchs += training_loss.item()

                    # Get the training prediction
                    training_prediction = get_output(model_output)

                    # Print a message after the training epoch
                    print('\t Training - Batch: {} - Loss: {} - '
                          'Prediction: {} - Ground Truth: {}'
                          .format(counter, training_loss, training_prediction,
                                  labels))

                    counter += 1

            # Add training loss for each epoch
            train_loss_per_epoch.append(train_loss_over_batchs/(counter-1))

            # Iterate over the validation data batch-wise
            proceed = True  # Continuation flag
            counter = 1  # Batch counter

            while proceed:

                # Get a new batch from the validation data generator
                batch, proceed = get_batch(validation_data, batch_size)

                # Check if the batch is non-empty
                if len(batch) > 0:

                    # Get the model input
                    image, labels, extended_labels = get_input(batch)

                    # Perform a single validation step
                    validation_loss, model_output = validate(image,
                                                             extended_labels)
                    val_loss_over_batchs += validation_loss.item()

                    # Get the validation prediction
                    validation_prediction = get_output(model_output)

                    # Print a message after the validation epoch
                    print('\t Validation - Batch: {} - Loss: {} - '
                          'Prediction: {} - Ground Truth: {}'
                          .format(counter, validation_loss,
                                  validation_prediction, labels))
                    counter += 1

            # Add validation loss for each epoch
            val_loss = val_loss_over_batchs/(counter-1)
            val_loss_per_epoch.append(val_loss)

            # save the model state dictionary if current val loss is lower
            if val_loss < min_val_loss:
                print(f'Saving model, current loss: {val_loss}, \
                      previous minimum loss: {min_val_loss}')
                save(self.architecture.state_dict(), Path(save_path,
                                                          'state_dict.pt'))
                min_val_loss = val_loss

        # Update and save the tracker
        self.tracker.update({'epochs': epoch,
                             'training_loss': train_loss_per_epoch,
                             'validation_loss': val_loss_per_epoch,
                             'model_state_dict': (
                                 self.architecture.state_dict()),
                             'optimizer_state_dict': (
                                 self.optimizer.state_dict())
                             })
        save(self.tracker, Path(save_path, 'tracker.pt'))

    def forward(
            self,
            image):
        """Perform a single forward pass through the model."""
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

        return model_output
