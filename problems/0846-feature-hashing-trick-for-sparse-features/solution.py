import hashlib

def feature_hash(features, n_features):
    vector = [0.0] * n_features

    for name, value in features.items():
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()

        index = int(digest[:8], 16) % n_features
        sign_value = int(digest[8:16], 16)

        sign = 1 if sign_value % 2 == 0 else -1

        vector[index] += sign * float(value)

    return vector