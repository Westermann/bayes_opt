# import numpy as np


class LandscapeModel(object):

    def __init__(self, regressor, kernel, kernel_args={}):
        self.kernel = kernel
        self.model = regressor(kernel=kernel(**kernel_args))

    def fit(self, Xy):
        X = Xy[:, :-1]
        y = Xy[:, -1:]
        return self.model.fit(X, y)

    def predict(self, x):
        # return self.model.predict(x.reshape(1, -1), return_std=True)
        return self.model.predict(x.reshape(1, -1))
