
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def correlation_metrics(tb, metrics):

    result_m = np.zeros((len(metrics), len(metrics)))

    for i, m1 in enumerate(metrics):

        v1 = tb[[m1]].values.flatten()

        for j, m2 in enumerate(metrics):

            v2 = tb[[m2]].values.flatten()

            result_m[i, j] = np.corrcoef(v1, v2)[1, 0]
            # result_m[i, j] = abs(np.corrcoef(v1, v2)[1, 0])
            
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



def plot_matrix(data, labels, plot_values=True, figsize=(12, 12)):


    # Create figure.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)


    # Plot matrix
    cax = ax.matshow(data, interpolation='nearest')


    # Color Bar
    bax = fig.colorbar(cax)
    bax.ax.tick_params(labelsize=20)


    # Configure and plot labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")


    # Plot values of the matrix
    if plot_values:

        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                
                if data[i, j] > 0.8: color = 'b'
                else: color = 'w'
                
                text = ax.text(j, i, '%.2f' % (data[i, j]),
                               ha="center", va="center", color=color, fontsize=15)

    #ax.set_title("Matrix comparing detection and tracking metrics.")

    plt.show()