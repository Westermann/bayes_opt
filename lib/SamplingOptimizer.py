import numpy as np
from .BayesianOptimizer import BayesianOptimizer


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