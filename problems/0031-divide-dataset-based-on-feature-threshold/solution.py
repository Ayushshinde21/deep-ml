import numpy as np

def divide_on_feature(X, feature_i, threshold):
    condition = X[:, feature_i] >= threshold

    left = X[condition]
    right = X[~condition]

    return [left, right]