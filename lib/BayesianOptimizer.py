import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize


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
        self.best_achieved = []
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
        try:
            self.best_achieved.append(np.max([target, self.best_achieved[-1]]))
        except IndexError:
            self.best_achieved.append(target)

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
        k = self.acquisition_params['k']
        mean, std = self.model.predict(x.reshape(1, -1), return_std=True)
        return -(mean + k * std)


class SamplingOptimizer(BayesianOptimizer):

    def __init__(self, feature_samples, **kw_args):
        """
        SamplingOptimizer
        Keyword arguments:
            feature_samples:
                Numpy feature matrix (ignored if `feature_meta` supplied)
            **kw_args:
                Keyword arguements for BayesianOptimizer
        """

        super().__init__(
            feature_samples=feature_samples,
            **kw_args
        )

    def suggest(self, return_dict=False):
        """
        Optimizes the acquisition function over the
        currently estimated parameter space sampling
        from the feature samples and returns the optimal
        suggested next sample.
        Keyword arguements:
            return_dict:
                Boolean indicating wether to return the
                optimum value as a { <feature_name>: <value> }
                dict (if `True`) or as a vector (default)

        """
        samples = self.feature_samples[
            np.random.choice(self.feature_samples.shape[0],
                             min(1000, self.feature_samples.shape[0]),
                             replace=False)]
        optimum_val = -np.inf
        for sample in samples:
            iter_val = self.acquisition(sample)
            print(sample)
            print(iter_val)
            if iter_val >= optimum_val:
                optimum_val = iter_val
                optimum = sample

        if return_dict is True:
            return dict(zip(self.feature_names, optimum))
        else:
            return optimum


class RangeOptimizer(BayesianOptimizer):

    def __init__(self, feature_meta, **optimizer_args):
        """
        RangeOptimizer
        Keyword arguments:
            feature_meta:
                Dictionary of
                { <feature_name>: ( <lower_bound>, <upper_bound> ) }
                The bounds should be of the type enabling the values
                this feature can take. E.g. int if it is a discrete
                parameter/feature and float if it is continuous
            **optimizer_args:
                Keyword arguements for BayesianOptimizer
        """

        super().__init__(
            feature_meta=feature_meta,
            **optimizer_args
        )

    def suggest(self, return_dict=False):
        """
        Optimizes the acquisition function over the
        currently estimated parameter landscape and
        returns the optimum value from the feature
        space.
        Keyword arguements:
            return_dict:
                Boolean indicating wether to return the
                optimum value as a { <feature_name>: <value> }
                dict (if `True`) or as a vector (default)

        """
        samples = np.random.uniform(
            self.feature_bounds[:, 0],
            self.feature_bounds[:, 1],
            size=(20, self.feature_bounds.shape[0]))
        optimum_val = -np.inf
        for sample in samples:
            opt_res = minimize(
                fun=self.acquisition,
                x0=sample,
                bounds=self.feature_bounds)
            if min(-opt_res.fun) >= optimum_val:
                optimum_val = min(-opt_res.fun)
                optimum = opt_res.x

        optimum = np.maximum(optimum, self.feature_bounds[:, 0])
        optimum = np.minimum(optimum, self.feature_bounds[:, 1])
        optimum = [t(optimum[i])
                   for i, t
                   in enumerate(self.feature_types)]
        if return_dict is True:
            return dict(zip(self.feature_names, optimum))
        else:
            return optimum
