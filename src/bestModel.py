
from logging import getLogger
from src.model import get_all_model
from sklearn.metrics import r2_score

logger = getLogger()


class ModelSelector:

    def __init__(self, loader, params):
        """
        Initialize imputation model selector
        """
        self.loader = loader
        self.params = params

    def train_models(self):

        models = get_all_model()
        for m in models:
            logger.info("Training model " + m.__str__())
            m.train(self.loader.x_train, self.loader.y_train)
            y_pred = m.predict(self.loader.x_test)
            score = r2_score(self.loader.y_test, y_pred)
            logger.info(m.__str__() + " training score is " + score.__str__())

