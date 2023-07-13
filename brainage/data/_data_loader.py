"""Data loading."""

# %% External package import

from os.path import splitext
from nibabel import load
from pandas import cut, read_csv
from sklearn.model_selection import StratifiedKFold
from warnings import filterwarnings
filterwarnings("ignore")

# %% Class definition


class DataLoader():
    """
    Data loading class.

    This class provides ...

    Parameters
    ----------
    data_path : string
        ...

    age_filter : list
        ...

    Attributes
    ----------
    age_filter : list
        ...

    sets : dict
        ...

    Methods
    -------
    - ``add_fold_numbers()`` : add a fold column for later \
        train-validate-test splitting;
    - ``split()`` : split into training and test data;
    - ``get_data(which)`` : get the raw data;
    - ``get_images(which)`` : get the images;
    - ``set_file_path(path, which)`` : set the image paths;
    - ``get_age_values(which)`` : get the age values;
    - ``set_age_values(values, which)`` : set the age values;
    - ``get_fold_numbers(which)`` : get the fold numbers;
    - ``set_fold_numbers(numbers, which)`` : set the fold numbers;
    - ``save_data_to_file(path)`` : write training and test subsets to csv \
        files.
    """

    def __init__(
            self,
            data_path,
            age_filter):

        print('\n\t Initializing the data loader ...')
        print('\t\t >>> Age filter: {} <<<'.format(age_filter))

        # Get the age filter from the arguments
        self.age_filter = age_filter

        try:

            print('\t\t Reading and modifying the input data ...')

            # Load the csv file as a dataframe
            self.sets = {'raw': read_csv(data_path, sep=','),
                         'train': None,
                         'test': None}

            # Check if age and file path are present as columns in the csv data
            if not all(column in self.sets['raw'].columns
                       for column in ('file_path', 'age')):
                raise ValueError("File must contain image paths and age "
                                 "values!")

            # Rename the values of file path
            self.sets['raw']['file_path'] = self.sets[
                'raw']['file_path'].str.replace(
                '.images', 'brainage/data/datasets/images')

            # Select samples based on the site
            self.sets['raw'] = self.sets['raw'][
                self.sets['raw']['site'] == 'IXI/IOP']

            # Select samples based on the given age range
            self.sets['raw'] = self.sets['raw'][
                self.sets['raw']['age'].between(age_filter[0], age_filter[1])]

            # Sort data based on age
            self.sets['raw'].sort_values(by='age', inplace=True,
                                         ignore_index=True)

            # Add the fold column
            self.add_fold_numbers()

            # Split into training and test data
            self.split()

            # Save training and test subsets to files
            self.save_data_to_file(data_path)

        except ValueError:

            # Create a dictionary with empty sets
            self.sets = {'raw': None,
                         'train': None,
                         'test': None}

    def add_fold_numbers(self):
        """Add a fold column for later train-validate-test splitting."""
        print('\t\t Adding the fold numbers ...')

        # Initialize the stratified k-fold object
        skf = StratifiedKFold(n_splits=5)

        # Generate age bins from the raw age values
        age_bins = cut(self.get_age_values('raw'), bins=5, precision=1,
                       labels=False)

        # Loop over the bin splits
        for val, (_, indices) in enumerate(skf.split(
                self.sets['raw'], age_bins)):

            # Assign the fold numbers to the corresponding indices
            self.sets['raw'].loc[indices, 'fold'] = int(val)

    def split(self):
        """Split into training and test data."""
        print('\t\t Splitting into training and test data ...')

        # Get the training data from all rows with fold number unequal to zero
        self.sets['train'] = self.sets['raw'][self.sets['raw']['fold'] != 0]

        # Get the test data from all rows with fold number equal to zero
        self.sets['test'] = self.sets['raw'][self.sets['raw']['fold'] == 0]

    def get_data(
            self,
            which='raw'):
        """Get the raw data."""
        return self.sets[which]

    def get_images(
            self,
            which='raw'):
        """Get the images."""
        return (load(path) for path in self.sets[which]['file_path'])

    def set_file_path(
            self,
            path,
            which):
        """Set the image paths."""
        self.sets[which]['file_path'] = path

    def get_age_values(
            self,
            which):
        """Get the age values."""
        return tuple(self.sets[which]['age'])

    def set_age_values(
            self,
            values,
            which):
        """Set the age values."""
        self.sets[which]['age'] = values

    def get_fold_numbers(
            self,
            which):
        """Get the fold numbers."""
        return tuple(self.sets[which]['fold'])

    def set_fold_numbers(
            self,
            numbers,
            which):
        """Set the fold numbers."""
        self.sets[which]['fold'] = numbers

    def save_data_to_file(
            self,
            path):
        """Write training and test subsets to csv files."""
        print('\t\t Saving data subsets to files ...')

        # Get the path name and the file extension
        name, extension = splitext(path)

        # Loop over the subset labels
        for subset in ('train', 'test'):

            # Save the subset as a csv file with the given file name
            self.sets[subset].to_csv(''.join((name, '_', subset, extension)),
                                     index=False)
