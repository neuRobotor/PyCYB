import numpy as np
import os
import csv
from scipy.stats import norm, chi
from convnet import summary
from convmemnet import data_proc
from convmemnet import train_net
from convmemnet import norm_emg
import pickle
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def ecdf_norm(emg_data, ecdfs):
    emg_data = norm_emg(emg_data)
    emg_data = np.abs(emg_data)
    #emg_data = norm_emg(emg_data)
    for chnl in range(len(emg_data)):
        a_max = 0.9999999999999999
        a_min = 0.0000000000000001
        transf = ecdfs[chnl](emg_data[chnl, :])
        transf = np.clip(transf, a_min=a_min, a_max=a_max)
        emg_data[chnl] = norm.ppf(transf)
    return emg_data

def train_n_time():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    overlap = 2
    freq_factor = 20
    n_channels = 8
    stride = 1
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]

    with open(data_path + r'\ecdfs.pickle', 'rb') as handle:
        ecdfs = pickle.load(handle)
    X, Y, files = data_proc(data_path, ecdf_norm, params=ecdfs, overlap=overlap, diff=diff,
                            window_size=freq_factor, n_channels=n_channels, task='Stair', stride=stride)

    print('Data loaded. Beginning training.')
    k = 5
    drop = 0.5
    kernel = (3, 3)
    dil = 0
    poolsize = 2
    ep, ba = 100, 128
    from convmemnet import  conv1d_model
    from functools import partial
    import json
    from convmemnet import stack_emg
    cur_model = partial(conv1d_model, n_timesteps=X.shape[1], n_features=X.shape[2],
                        n_outputs=Y.shape[1], drp=drop, krnl=kernel, dilate=dil, mpool=poolsize)
    model = cur_model()
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    X0 = np.empty((0, overlap * freq_factor, n_channels))
    Y0 = [[] for _ in range(len(joint_names))]
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
        if 'Stair36' not in file:
            continue
        files.append(file)
        with open(data_path + '\\' + file) as json_file:
            dict_data = json.load(json_file)
        for i, joint in enumerate(joint_names):
            cur_data = dict_data[joint]
            Y0[i].extend(cur_data[(overlap - 1):])
        emg_data = ecdf_norm(np.array(dict_data["EMG"]), ecdfs)
        emg_data = stack_emg(emg_data, freq_factor, overlap)
        X0 = np.vstack((X0, emg_data))
    Y0 = np.array(Y0)
    Y0 = Y0[:, :, 0].transpose()

    model.fit(X, Y, batch_size=ba, epochs=ep, verbose=2, callbacks=None, validation_data=(X0, Y0))
    import time

    start = time.time()
    for _ in range(10):
        Y1 = model.predict(X0)
    end = time.time()
    print(end - start)
    import matplotlib.pyplot as plt
    plt.plot(Y1)
    plt.show()

def kfold():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    overlap = 2
    freq_factor = 20
    n_channels = 8
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]
    stride = 1
    with open(data_path + r'\ecdfs.pickle', 'rb') as handle:
        ecdfs = pickle.load(handle)

    X, Y, files = data_proc(data_path, ecdf_norm, params=ecdfs, overlap=overlap, diff=diff,
                            window_size=freq_factor, n_channels=n_channels, task='Stair', stride=stride)
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
    #train_n_time()
    pass


if __name__ == "__main__":
    main()

