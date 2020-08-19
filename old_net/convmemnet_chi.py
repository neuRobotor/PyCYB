import numpy as np
import os
import csv
from scipy.stats import norm, chi
from Archive.convnet import summary
from Archive.convmemnet import data_proc
from Archive.convmemnet import train_net
from Archive.convmemnet import norm_emg
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def chi_norm(emg_data, params):
    emg_data = norm_emg(emg_data)
    emg_data = np.abs(emg_data)
    #emg_data = norm_emg(emg_data)
    for chnl in range(len(emg_data)):
        arg = params[chnl][:-2]
        loc = params[chnl][-2]
        scale = params[chnl][-1]
        a_max = chi.ppf(0.9999999999999999, loc=loc, scale=scale, *arg)
        a_min = chi.ppf(0.00000000001, loc=loc, scale=scale, *arg)
        transf = np.clip(emg_data[chnl, :], a_min=a_min, a_max=a_max)
        transf = chi.cdf(transf, loc=loc, scale=scale, *arg)
        emg_data[chnl] = norm.ppf(transf)
    return emg_data


def kfold():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    overlap = 2
    freq_factor = 20
    n_channels = 8
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]

    with open(data_path+r'\params.csv') as f:
        reader = csv.reader(f)
        params = list(reader)
        params = [[float(el) for el in row] for row in params]

    X, Y, files = data_proc(data_path, chi_norm, params=params, overlap=overlap, diff=diff,
                            window_size=freq_factor, n_channels=n_channels, task='Stair')
    print('Data loaded. Beginning training.')

    # ########################
    #       TRAIN NETWORK
    # ########################

    k = 5
    drop = 0.5
    kernel = (3, 3)
    dil = 0
    poolsize = 2
    ep, ba = 100, 128
    scores, model = train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba)
    summary(k, scores, kernel, drop, model, data_path, ep, ba, files)


def main():
    kfold()
    pass


if __name__ == "__main__":
    main()

