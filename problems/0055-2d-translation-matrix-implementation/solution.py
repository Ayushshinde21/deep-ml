import numpy as np

def translate_object(points, tx, ty):
    points = np.array(points, dtype=float)

    translated_points = points + np.array([tx, ty])

    return translated_points.tolist()