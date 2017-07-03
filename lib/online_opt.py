import numpy as np
from .exceptions import FeatureConfigurationMissingError


class OnlineOptimizer(object):

    def __init__(self, feature_meta=None, feature_samples=None):
        if feature_meta is None and feature_samples is None:
            raise FeatureConfigurationMissingError()

        if feature_meta is not None:
            self.feature_names = feature_meta.keys()
            self.feature_bounds = np.stack(feature_meta.values())
            self.feature_types = [type(f[0]) for f in feature_meta.values()]
            self.features_dim = len(self.feature_names)

        if feature_samples is not None:
            self.feature_samples = feature_samples
            self.features_dim = len(feature_samples[0])

        self.i = 0
        self.observations = []
        self.best_achieved = []

    def update_meta(self, features, target):
        if type(features) == dict:
            features = [features[f] for f in self.feature_names]

        self.i += 1
        self.observations.append(features + [target])
        try:
            self.best_achieved.append(np.max([target, self.best_achieved[-1]]))
        except IndexError:
            self.best_achieved.append(target)

        return features, target

    def get_best_features(self):
        observations = np.array(self.observations)
        best_features = observations[np.argmax(observations[:, -1]), :-1]
        try:
            best_features = dict(zip(self.feature_names, best_features))
        except TypeError:
            pass

        return best_features
