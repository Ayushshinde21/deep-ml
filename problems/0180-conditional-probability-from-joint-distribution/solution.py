def conditional_probability(joint_distribution: dict) -> float:

    p_ab = joint_distribution[('A', 'B')]
    p_b = joint_distribution[('A', 'B')] + joint_distribution[('`A', 'B')]

    if p_b == 0:
        return 0.0

    return round(p_ab / p_b, 4)