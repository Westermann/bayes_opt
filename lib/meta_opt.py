import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from .online_opt import OnlineOptimizer
from .bayes_opt import RangeOptimizer, SamplingOptimizer
from .exceptions import FeatureConfigurationMissingError


class MetaOptimizer(OnlineOptimizer):

    def __init__(self,
                 probation=5,
                 feature_meta=None,
                 feature_samples=None,
                 **optimizer_args):
        super().__init__(feature_meta=feature_meta,
                         feature_samples=feature_samples)
        if feature_meta is not None:
            self.Optimizer = RangeOptimizer
            optimizer_features_arg = {
                'feature_meta': feature_meta
            }
        elif feature_samples is not None:
            self.Optimizer = SamplingOptimizer
            optimizer_features_arg = {
                'feature_samples': feature_samples
            }
        else:
            raise FeatureConfigurationMissingError()

        self.experts = []
        for kernel in [
            Matern(nu=1/2),
            Matern(nu=3/2),
            Matern(nu=5/2),
            RBF(),
            RationalQuadratic()
        ]:
            opt = self.Optimizer(**{
                **optimizer_features_arg,
                **optimizer_args,
                'kernel': kernel
            })
            self.experts.append({
                'name': str(opt),
                'opt': opt
            })

        self.probation = probation
        self.i = 0

    def update(self, features, target):
        """
        Receives a feature vector and a target value
        and refits all experts' gaussian process estimations
        for the parameter surrogate. Returns the optimizer(self).
        The target values are maximised!
        """
        features, target = super().update_meta(features, target)
        [e['opt'].update(features, target) for e in self.experts]
        return self

    def suggest(self, return_dict=False):
        """
        Optimizes the acquisition function over the
        currently estimated parameter surrogate  of all
        experts and returns the optimum value from the feature
        space for the expect that achieves the highest
        'trustworhiness'.
        Keyword arguements:
            return_dict:
                Boolean indicating wether to return the
                optimum value as a { <feature_name>: <value> }
                dict (if `True`) or as a vector (default)
        """
        if self.i > self.probation:
            lls = [
                e['opt'].model.log_marginal_likelihood_value_
                for e in self.experts
            ]
            x = self.experts[np.argmax(lls)]['opt'].suggest(return_dict=return_dict)
        else:
            x = np.random.choice(self.experts)['opt'].suggest(return_dict=return_dict)

        return x
