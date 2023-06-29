"""Data loading."""

# %% External package import

from os.path import splitext
import nibabel as nib
from pandas import cut, read_csv
from sklearn.model_selection import StratifiedKFold

# %% Internal package import

# %% Class definition


class DataLoader():
    """
    Data loading class.

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
            data_path,
            age_filter):

        # Get the age filter from the arguments
        self.age_filter = age_filter

        # Load the csv file as a dataframe
        self.sets = {'raw': read_csv(data_path, sep=','),
                     'train': None,
                     'test': None}

        # Check if age and file path are present as columns in the csv data
        if not all(column in self.sets['raw'].columns
                   for column in ('file_path', 'age')):
            raise ValueError("File must contain image paths and age values!")

        # Rename the values of file path
        self.sets['raw']['file_path'] = self.sets[
            'raw']['file_path'].str.replace(
            '.images', 'brainage/data/datasets/images')

        # Select samples based on the site
        self.sets['raw'] = self.sets['raw'][
            self.sets['raw']['site'] == 'IXI/IOP']

        # Select samples based on the given age range
        self.sets['raw'] = self.sets['raw'][self.sets['raw']['age'].between(
            age_filter[0], age_filter[1])]

        # Sort data based on age
        self.sets['raw'].sort_values(by='age', inplace=True, ignore_index=True)

        # Add the fold column
        self.add_fold_numbers()

        # Split into training and test data
        self.split()

        # Save training and test subsets to files
        self.save_data_to_file(data_path)

    def add_fold_numbers(self):
        """Add a fold column for later train-validate-test splitting."""
        skf = StratifiedKFold(n_splits=5)
        age_bins = cut(self.get_age_values('raw'), bins=5, precision=1,
                       labels=False)

        for val, (_, validation_indices) in enumerate(skf.split(
                self.sets['raw'], age_bins)):
            self.sets['raw'].loc[validation_indices, 'fold'] = int(val)

        self.sets['raw']['fold'] = self.sets['raw']['fold'].astype(int)

    def split(self):
        """Split into training and test data."""
        self.sets['train'] = self.sets['raw'][self.sets['raw']['fold'] != 0]
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
        return (nib.load(path) for path in self.sets[which]['file_path'])

    def set_file_path(
            self,
            path,
            which):
        """Set the images."""
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
        name, extension = splitext(path)
        train_path = ''.join((name, '_train', extension))
        test_path = ''.join((name, '_test', extension))

        self.sets['train'].to_csv(train_path, index=False)
        self.sets['test'].to_csv(test_path, index=False)
