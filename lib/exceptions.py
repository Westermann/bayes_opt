class FeatureConfigurationMissingError(Exception):
    """
    Raise when any optimizer is initialized
    without proper feature declarations.
    """
    def __init__(self):
        message = 'Please provide at least one of ' + \
                  '`feature_meta` or `feature_samples`'
        super().__init__(message)
