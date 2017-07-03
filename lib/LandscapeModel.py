import numpy as np
import george as sklearn_gp
from sklearn import gaussian_process as sklearn_gp


class LandscapeModel(object):

    def __init__(self, lib):
        self.regressor = GaussianProcessRegressor(
