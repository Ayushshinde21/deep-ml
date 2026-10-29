import numpy as np

def ordinal_encode(X, categories):
    category_maps = [
        {category: index for index, category in enumerate(column_categories)}
        for column_categories in categories
    ]

    encoded = [
        [category_maps[j].get(value, -1) for j, value in enumerate(row)]
        for row in X
    ]

    return np.array(encoded, dtype=int).reshape(len(X), len(categories))