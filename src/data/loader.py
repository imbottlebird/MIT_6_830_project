import pandas as pd
from logging import getLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = getLogger()

class Loader:

    def __init__(self, params, data=None):
        """
        Initialize data loader
        """
        self.params = params
        self.data = data
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.testing_data = {}
        self.cols_with_MV = None

    def load_data(self):

        self.data = pd.read_csv(self.params.dataset)
        logger.info('Data loaded in loader')

    def identify_missing_values(self):
        """
        Identify indices of columns with missing values (MV)
        """
        # assume single missing value column for now
        self.cols_with_MV = self.data.columns[self.data.isnull().any()].tolist()

    def set_train_test_data(self):
        """
        Get and split training and testing data
        """
        full_data = self.data.dropna()
        x = full_data.drop(self.cols_with_MV, axis=1)
        y = full_data[self.cols_with_MV]
        assert(x.shape[0] == y.shape[0])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)

    def transform_categorical(self):
        # hard code for now
        self.data['class'] = LabelEncoder().fit_transform(self.data['class'])

