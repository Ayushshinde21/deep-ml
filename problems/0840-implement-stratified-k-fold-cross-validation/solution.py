import numpy as np

def stratified_kfold_indices(y, n_splits):
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)

    if n_splits < 2 or n_splits > counts.min():
        raise ValueError("n_splits must be between 2 and the smallest class size")

    n_samples = len(y)
    fold_test_indices = [[] for _ in range(n_splits)]

    for label in labels:
        class_indices = np.where(y == label)[0].tolist()
        class_size = len(class_indices)
        base_size = class_size // n_splits
        remainder = class_size % n_splits
        start = 0

        for fold in range(n_splits):
            group_size = base_size + (1 if fold < remainder else 0)
            fold_test_indices[fold].extend(
                class_indices[start:start + group_size]
            )
            start += group_size

    all_indices = set(range(n_samples))
    result = []

    for test_indices in fold_test_indices:
        test_indices = sorted(test_indices)
        train_indices = sorted(all_indices - set(test_indices))
        result.append([train_indices, test_indices])

    return result