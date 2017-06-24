import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class SamplingBayesianOptimizer(object):

    def __init__(self, feature_samples,
                 init_observations=[],
                 kernel=Matern(nu=.5)):
        self.feature_samples = feature_samples
        # self.feature_names = feature_meta.keys()
        # self.feature_bounds = np.stack(feature_meta.values())
        # self.feature_types = [type(f[0]) for f in feature_meta.values()]
        # self.features_dim = len(self.feature_names)
        self.observations = [init_observations]
        self.i = 0
        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.acquisition_params = {
            'type': 'ucb',
            'u': 2.5   # TODO: this is an arbitrary value
        }
        self.acquisition = self._ucb

    def update(self, features, target):
        self.observations.append(features + [target])
        data = np.array(self.observations)
        X = data[:, :-1]
        y = data[:, -1:]
        self.model.fit(X, y)
        return self

    def suggest(self, return_dict=False):
        self.i += 1
        samples = self.\
            feature_samples[
                np.random.randint(self.feature_samples.shape[0], size=100),
                :]
        optimum_val = -np.inf
        for sample in samples:
            opt_res = minimize(
                fun=self.acquisition,
                x0=sample)
            iter_optimum_val = np.min(-opt_res.fun)
            if iter_optimum_val >= optimum_val:
                optimum_val = iter_optimum_val
                optimum = opt_res.x

        # self.feature_samples = np.stack([r for r
        #                                  in self.feature_samples
        #                                  if (r != optimum).any()])
        # optimum = np.maximum(optimum, self.feature_bounds[:, 0])
        # optimum = np.minimum(optimum, self.feature_bounds[:, 1])
        # optimum = [t(optimum[i])
        #            for i, t
        #            in enumerate(self.feature_types)]
        if return_dict is True:
            return dict(zip(self.feature_names, optimum))
        else:
            return optimum

    def step(self, features, target):
        self.update(features, target)
        return self.suggest()

    def _ucb(self, x):
        u = self.acquisition_params['u']
        mean, std = self.model.predict(x.reshape(1, -1), return_std=True)
        result = -(mean + u * std).reshape(1)
        return result
