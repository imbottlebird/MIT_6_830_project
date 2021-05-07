from sklearn import tree


class DecisionTree:

    def __init__(self):
        """
        Initialize Decision Tree object.
        """
        self.model = tree.DecisionTreeRegressor()

    def __str__(self):
        return self.model.__str__()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

