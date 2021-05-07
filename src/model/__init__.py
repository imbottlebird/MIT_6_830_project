from src.model.decisionTree import DecisionTree

all_model_names = ["decisionTree"]


def build_imputation_model(params):
    """
    Build imputation model.
    """
    pass


def get_all_model():
    models = []
    for name in all_model_names:
        model = get_model(name)
        models.append(model)
    return models


def get_model(name):
    if name == "decisionTree":
        dt = DecisionTree()
        return dt
