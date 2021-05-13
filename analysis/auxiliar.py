
import pandas as pd
import numpy as np

from scipy.spatial import distance


def correlation_matrix(tb, metrics='all', dist_func=distance.correlation):
    
    comb = combinations(tb)
    result_m = np.zeros((len(comb), len(comb)))
    
    if metrics == 'all':         func_metrics = all_metrics
    elif metrics == 'tracking':  func_metrics = tracking_metrics
    elif metrics == 'detection': func_metrics = detection_metrics


    for i, (tr1, dt1) in enumerate(comb):

        m1 = search(tb, tr1, dt1)
        m1 = func_metrics(m1)

        for j, (tr2, dt2) in enumerate(comb):

            m2 = search(tb, tr2, dt2)
            m2 = func_metrics(m2)

            aux_m = []

            for v1, v2 in zip(m1.values, m2.values):

                aux_m.append( dist_func(v1, v2) )


            result_m[i, j] = 1 - (sum(aux_m) / len(aux_m))
            
    return result_m


def combinations(tb):

    trck = pd.unique(tb['Tracker'])
    detc = pd.unique(tb['Detector'])


    combin = []

    for tr in trck:

        for dc in detc:

            combin.append((tr, dc))


    return combin



def search(tb, tracker, detector):

    return tb.loc[(tb['Tracker'] == tracker) & (tb['Detector'] == detector)]



def tracking_metrics(tb):

    return tb.iloc[:, 11:]


def detection_metrics(tb):

    return tb.iloc[:, 4:11]


def all_metrics(tb):

    return tb.iloc[:, 4:]