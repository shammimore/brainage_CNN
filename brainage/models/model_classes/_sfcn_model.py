"""SFCN model."""

# %% External package import

import itertools
from numpy import exp
from torch import as_tensor, device, float32, load, no_grad
from torch.nn import Conv3d, DataParallel, init, Module
from torch.optim import Adam, SGD

# %% Internal package import

from brainage.models.architectures import SFCN
from brainage.tools import get_bin_centers, num2vect
from brainage.models.loss_functions import KLDivLoss

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
        ...

    age_filter : ...
        ...

    architecture : ...
        ...

    Methods
    -------
    - ``freeze_inner_layers()`` : ...
    - ``adapt_output_layer(age_range)`` : ...
    - ``set_optimizer(optimizer, learning_rate)`` : ...
    - ``fit(data, number_of_epochs, batch_size)`` : ...
    - ``forward(image)`` : ...
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

        # Initialize the model architecture
        self.architecture = SFCN()

        # Parallelize the model
        self.architecture = DataParallel(self.architecture)

        # Load the pretrained weights if applicable
        if pretrained_weights:
            self.architecture.load_state_dict(load(
                pretrained_weights, map_location=device(comp_device)))

    def freeze_inner_layers(self):
        """Freeze the parameters of the input and hidden layers."""
        # Get all and the output layer parameters
        all_params = self.parameters()
        out_params = self.architecture.module.classifier.conv_6.parameters()

        # Set the gradient calculation for all parameters to False
        for param in all_params:
            param.requires_grad = False

        # Set the gradient calculation for the output layer to True
        for param in out_params:
            param.requires_grad = True

    def adapt_output_layer(
            self,
            age_range):
        """Adapt the output layer for the age range."""
        self.module.classifier.conv_6 = Conv3d(64, age_range, padding=0,
                                               kernel_size=1)
        init.kaiming_normal_(self.module.classifier.conv_6.weight)

    def set_optimizer(
            self,
            optimizer,
            learning_rate):
        """Set the optimizer for the model training."""
        if optimizer == 'adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad,
                                         self.parameters()),
                                  lr=learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0,
                                  amsgrad=False)
        elif optimizer == 'sgd':
            self.optimizer = SGD(filter(lambda p: p.requires_grad,
                                        self.parameters()),
                                 lr=learning_rate,
                                 momentum=0.9,
                                 weight_decay=0.001)

    def fit(
            self,
            data,
            number_of_epochs,
            batch_size):
        """Fit the SFCN model."""
        print('This is where the fitting will be done - someday ... someday is today')        
        
        def get_input(image_label_data):
            bin_step = 1
            sigma = 1
            image = image_label_data[0]
            label = image_label_data[1]

            # Transforming the age to soft label (probability distribution)
            # label = label.numpy().reshape(-1)
            y, centers = num2vect(label, self.age_filter, bin_step, sigma) # probabilities, bin centers
            y = as_tensor(y, dtype=float32, device=self.comp_device)

            # Get the image shape
            dims = image.shape
            print('dims', dims)
            c, d, h, w = image.shape 
            b = 1 # remove once we load a batch
            image = image.reshape(b, dims[0], dims[1], dims[2], dims[3]) # remove once we load a batch
            image = as_tensor(image, dtype=float32, device=self.comp_device)
            return image, y, centers

        def get_output(y, centers, out):
            out = out.detach().cpu().numpy()
            y = y.cpu().numpy()
            prob = exp(out)
            pred = prob @ centers
            print('prediction: ', pred)
            return pred

        def train(image):
            print('Training mode')
            b = 1
            self.architecture.train()
            self.optimizer.zero_grad()
            output = self.architecture(image)
            out = output[0].reshape([b, -1])
            loss = KLDivLoss(out, y) 
            loss.backward()
            self.optimizer.step() 
            # train_loss += loss.item()
            return loss, out

        def validate(image):
            print('Evaluation mode')
            b = 1
            self.architecture.eval()
            with no_grad():
                output = self.architecture(image)
            out = output[0].reshape([b, -1])
            loss = KLDivLoss(out, y)
            # val_loss += loss.item()
            return loss, out
        
        # create training and validation data
        train_generator, validate_generator = itertools.tee(data)
        training_data = (sample for sample in train_generator 
                        if sample[2] != 1)
        validation_data = (sample for sample in validate_generator 
                        if sample[2] == 1)
             
        
        print('--CURRENT STATE: only runs for one epoch, probably because of the property of iterator')   
        for epoch in range(number_of_epochs):
            print('\n----Epoch number: %d------' %(epoch))
            
            # iterate over training data
            for i, train_image_label in enumerate(training_data):
                print('Sample number: %d' %(i))
                image, y, centers = get_input(train_image_label)
                loss, out = train(image)
                pred = get_output(y, centers, out)
                
            # iterate over validation data
            for i, validate_image_label in enumerate(validation_data): 
                print('Sample number: %d' %(i))
                image, y, centers = get_input(validate_image_label)
                loss, out = validate(image)
                pred = get_output(y, centers, out)
            
        


    def forward(
            self,
            image):
        """Perform a single forward pass through the model."""
        # Get the bin centers
        centers = get_bin_centers(self.age_filter, 1)

        # Get the image shape
        dims = image.shape

        # Reshape the image to 5D (with batch size 1)
        image = image.reshape(1, dims[0], dims[1], dims[2], dims[3])

        # Convert the image to a tensor
        image = as_tensor(image, dtype=float32, device=self.comp_device)

        # Set the architecture into evaluation mode (affects BatchNorm)
        self.architecture.eval()

        # Get the model output with gradient calculation disabled
        with no_grad():
            output = self.architecture(image)

        # Shift the output back to the CPU
        out = output[0].cpu().reshape([1, -1])

        # Convert the output to a numpy array
        out = out.numpy()

        # Calculate the probabilities
        prob = exp(out)

        # Get the prediction with the bin centers
        pred = prob @ centers

        return pred
