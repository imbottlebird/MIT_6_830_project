

all_model_names = ["decisionTree"]


def build_imputation_model(params):
    """
    Build imputation model.
    """
    if params.imputation_only:
        from .decisionTree import build_decision_tree
        return build_decision_tree()


def get_all_model():
    models = []
    for name in all_model_names:
        model = get_model(name)
        models.append(model)
    return models


def get_model(name):
    if name == "decisionTree":
        from.decisionTree import build_decision_tree
        return build_decision_tree()