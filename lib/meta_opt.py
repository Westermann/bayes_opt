from bayes_opt import RangeOptimizer, SamplingOptimizer

class MetaOptimizer(object):

    def __init__(self,
                 feature_meta=None,
                 feature_samples=None,
                 **optimizer_args):
        if feature_meta is not None:
            self.Optimizer = RangeOptimizer
        elif feature_samples is not None:
            self.Optimizer = SamplingOptimizer
        else:
            raise Exception('Please provide at least one of ' +
                            '`feature_meta` or `feature_samples`')
