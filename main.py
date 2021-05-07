
import argparse

from src.utils import bool_flag
from src.model import build_imputation_model
from src.data.loader import Loader
from src.train import Trainer
from src.bestModel import ModelSelector

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='Data imputation')

    # model selection mode
    parser.add_argument("--model_selection", type=bool_flag, default=True,
                        help="Selection of best imputation model")

    # imputation mode
    parser.add_argument("--imputation_only", type=bool_flag, default=False,
                        help="Only do imputation true or false")

    # data set
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset to be imputed in relative path")

    return parser


def main(params):

    loader = Loader(params)
    loader.load_data()

    if params.model_selection:
        loader.transform_categorical()
        loader.identify_missing_values()
        loader.set_train_test_data()
        selector = ModelSelector(loader, params)
        selector.train_models()
    else:
        pass


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args("--dataset ./datasets/iris_with_MV.csv --model_selection True".split())
    main(params)
