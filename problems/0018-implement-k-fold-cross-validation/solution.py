import numpy as np

def k_fold_cross_validation(n_samples, k=5, shuffle=True):
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    fold_sizes = [n_samples // k] * k

    for i in range(n_samples % k):
        fold_sizes[i] += 1

    folds = []
    start = 0

    for size in fold_sizes:
        folds.append(indices[start:start + size].tolist())
        start += size

    splits = []

    for i in range(k):
        test_indices = folds[i]
        train_indices = []

        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])

        splits.append((train_indices, test_indices))

    return splits