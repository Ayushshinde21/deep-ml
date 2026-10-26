def label_encode_ordinal(values: list, order: list) -> list:
    mapping = {category: i for i, category in enumerate(order)}
    return [mapping.get(value, -1) for value in values]