import numpy as np
import os
import csv
from Archive.convnet import summary
from Archive.convmemnet import data_proc
from Archive.convmemnet import train_net


#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def box_cox(data, lam):
    if not lam:
        return np.log(data)
    return (np.power(data, lam)-1)/lam


def box_cox_norm(emg_data, params):
    emg_data = np.abs(emg_data)
    for chnl in range(len(emg_data)):
        emg_data[chnl] = box_cox(emg_data[chnl], params[chnl])
    emg_std = np.std(emg_data, axis=1)
    emg_mean = np.mean(emg_data, axis=1)
    return (emg_data - emg_mean[:, None]) / emg_std[:, None]


def kfold():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    overlap = 2
    freq_factor = 20
    n_channels = 8
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]

    with open(data_path+r'\lambdas.csv') as f:
        reader = csv.reader(f)
        params = list(reader)
        params = [[float(el) for el in row] for row in params]

    X, Y, files = data_proc(data_path, box_cox_norm, params=params, overlap=overlap, diff=diff,
                            window_size=freq_factor, n_channels=n_channels, task='Stair')

    print('Data loaded. Beginning training.')

    # ########################
    #       TRAIN NETWORK
    # ########################

    k = 5
    drop = 0.5
    kernel = (3, 3)
    dil = 3
    poolsize = 2
    ep, ba = 100, 64
    scores, model = train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba)
    summary(k, scores, kernel, drop, model, data_path, ep, ba, files)


def main():
    kfold()
    pass


if __name__ == "__main__":
    main()

