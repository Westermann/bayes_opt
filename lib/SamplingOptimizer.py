import numpy as np
from .BayesianOptimizer import BayesianOptimizer
from scipy.optimize import minimize


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
        samples = self.\
            feature_samples[
                np.random.randint(
                    self.feature_samples.shape[0],
                    size=100), :]
        optimum_val = -np.inf
        for sample in samples:
            opt_res = minimize(
                fun=self.acquisition,
                x0=sample)
            iter_optimum_val = np.min(-opt_res.fun)
            if iter_optimum_val >= optimum_val:
                optimum_val = iter_optimum_val
                optimum = opt_res.x

        if return_dict is True:
            return dict(zip(self.feature_names, optimum))
        else:
            return optimum
