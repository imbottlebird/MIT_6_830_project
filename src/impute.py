
from logging import getLogger

logger = getLogger()

class Impute():

    def __init__(self, model, data, params):
        """
        Initialize impute.
        """
        self.model = model
        self.data = data
        self.params = params




