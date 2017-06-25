import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer(object):

    def __init__(self, feature_meta=None,
                 feature_samples=None,
                 init_observations=[],
                 kernel=Matern(nu=.5),
                 acquisition_params=None):
        """
        BayesianOptimizerSuperClass
        Keyword arguments:
            feature_meta:
                Dictionary of
                { <feature_name>: ( <lower_bound>, <upper_bound> ) }
                The bounds should be of the type enabling the values
                this feature can take. E.g. int if it is a discrete
                parameter/feature and float if it is continuous
            feature_samples:
                Numpy feature matrix (ignored if `feature_meta` supplied)
            init_observations:
                List of tuples of observation vectors with length
                `number_of_features + 1` where the last column is the
                target value
            kernel:
                A kernel function from sklearn.gaussian_process.kernel
                or equivalent. Default is `Matern(nu=.5)`
        """

        if feature_meta is None and feature_samples is None:
            raise Exception('Please provide at least one of ' +
                            '`feature_meta` or `feature_samples`')

        if feature_meta is not None:
            self.feature_names = feature_meta.keys()
            self.feature_bounds = np.stack(feature_meta.values())
            self.feature_types = [type(f[0]) for f in feature_meta.values()]
            self.features_dim = len(self.feature_names)

        if feature_samples is not None:
            self.feature_samples = feature_samples
            self.features_dim = len(feature_samples[0])

        self.observations = init_observations
        self.i = 0
        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.acquisition_params = acquisition_params
        if self.acquisition_params is None:
            self.acquisition_params = {
                'type': 'upper_confidence_bound',
                'k': .5   # TODO: this is an arbitrary value
            }

        self.acquisition = self._upper_confidence_bound

    def update(self, features, target):
        """
        Receives a feature vector and a target values
        and refits the gaussian process estimation for
        the parameter landscape. Returns the optimizer(self).
        The target values are maximised!
        """
        self.i += 1
        self.observations.append(features + [target])
        data = np.array(self.observations)
        X = data[:, :-1]
        y = data[:, -1:]
        self.model.fit(X, y)
        return self

    def step(self, features, target):
        self.update(features, target)
        return self.suggest()

    def _upper_confidence_bound(self, x):
        """
        Receives a sample and returns the upper confidence value
        for that point in the parameter/feature space
        """
        targets = np.array(self.observations)[:, -1]
        adapter = np.max(targets) - np.mean(targets)\
            / np.max(targets)
        k = self.acquisition_params['k'] * (1 / (self.i + 1)) * adapter
        mean, std = self.model.predict(x.reshape(1, -1), return_std=True)
        return -(mean + k * std)
