import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    np.random.seed(random_seed)

    X = np.asarray(X)
    y = np.asarray(y)

    train_indices = []
    test_indices = []

    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        np.random.shuffle(class_indices)

        n_test = int(len(class_indices) * test_size)

        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test