import numpy as np
import pandas as pd
def descriptive_statistics(data: list | np.ndarray) -> dict:
    mn = np.mean(data)
    md = np.median(data)
    mo = pd.Series(data).mode()[0]
    vr = np.var(data)
    st = np.std(data)
    q1 = np.percentile(data,25)
    q2 = np.percentile(data,50)
    q3 = np.percentile(data,75)
    iqr = q3 - q1

    x = {'mean': mn, 'median': md, 'mode': mo, 'variance': vr, 'standard_deviation':st ,'25th_percentile': q1, '50th_percentile': q2, '75th_percentile': q3, 'interquartile_range': iqr }

    return x