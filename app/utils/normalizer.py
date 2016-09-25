import numpy as np

# Implement the scikit learn API for our custom z-score normalizer
class NormalizePositive(object):
    def __init__(self, axis=0):
        self.axis = axis

    # compute the mean and standard deviation of the values that are not zero.
    # zeros indicate missing values...
    def fit(self, features, y=None):
        if self.axis == 1:
            features = features.T
        # count features that are greater than zero in axis 0:
        binary = (features > 0)
        count0 = binary.sum(axis=0)

        # to avoid division by zero, set zero counts to one:
        count0[count0 == 0] = 1.

        # computing the mean is easy:
        self.mean = features.sum(axis=0) / count0

        # only consider differences where binary is True:
        diff = (features - self.mean) * binary
        diff **= 2

        # regularize the estimate of std by adding 0.1
        # We add 0.1 to the direct estimate of the standard deviation to avoid underestimating
        # the value of the standard deviation when there are only a few samples, all of which
        # may be exactly the same. The exact value used does not matter much for the final
        # result, but we need to avoid division by zero.
        self.std = np.sqrt(0.1 + diff.sum(axis=0) / count0)
        return self

    def transform(self, features):
        if self.axis == 1:
            features = features.T
        binary = (features > 0)
        features = features - self.mean
        features /= self.std
        features *= binary
        #  transformed it back so that the return value has the same shape as the input
        if self.axis == 1:
            features = features.T
        return features

    def inverse_transform(self, features, copy=True):
        if copy:
            features = features.copy()
        if self.axis == 1:
            features = features.T
        features *= self.std
        features += self.mean
        if self.axis == 1:
            features = features.T
        return features

    def fit_transform(self, features):
        return self.fit(features).transform(features)