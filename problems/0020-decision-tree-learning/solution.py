import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    counts = Counter(labels)
    total = len(labels)

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    labels = [example[target_attr] for example in examples]

    total_entropy = calculate_entropy(labels)
    total = len(examples)

    values = sorted(set(example[attr] for example in examples))

    weighted_entropy = 0.0

    for value in values:
        subset = [
            example for example in examples
            if example[attr] == value
        ]

        subset_labels = [example[target_attr] for example in subset]

        weighted_entropy += (
            len(subset) / total
        ) * calculate_entropy(subset_labels)

    return total_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    counts = Counter(example[target_attr] for example in examples)
    max_count = max(counts.values())

    classes = [
        label for label, count in counts.items()
        if count == max_count
    ]

    return sorted(classes)[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    labels = [example[target_attr] for example in examples]

    if len(set(labels)) == 1:
        return labels[0]

    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = attributes[0]
    best_gain = calculate_information_gain(
        examples, best_attr, target_attr
    )

    for attr in attributes[1:]:
        gain = calculate_information_gain(
            examples, attr, target_attr
        )

        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    tree = {best_attr: {}}

    values = sorted(set(example[best_attr] for example in examples))

    remaining_attributes = [
        attr for attr in attributes
        if attr != best_attr
    ]

    for value in values:
        subset = [
            example for example in examples
            if example[best_attr] == value
        ]

        tree[best_attr][value] = learn_decision_tree(
            subset,
            remaining_attributes,
            target_attr
        )

    return tree