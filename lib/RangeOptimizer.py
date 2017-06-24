import numpy as np
from .BayesianOptimizer import BayesianOptimizer
from scipy.optimize import minimize


class RangeOptimizer(BayesianOptimizer):

    def __init__(self, feature_meta, **kw_args):
        """
        SamplingOptimizer
        Keyword arguments:
            feature_meta:
                Dictionary of
                { <feature_name>: ( <lower_bound>, <upper_bound> ) }
                The bounds should be of the type enabling the values
                this feature can take. E.g. int if it is a discrete
                parameter/feature and float if it is continuous
            **kw_args:
                Keyword arguements for BayesianOptimizer
        """

        super().__init__(
            feature_meta=feature_meta,
            **kw_args
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
            size=(1000, self.feature_bounds.shape[0]))
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
