import numpy as np
from scipy.stats import norm, chi
import os
import csv
import json
import multiprocessing
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import power_transform


def parallel_proc(data):
    # data = np.abs(data[abs(data - np.mean(data)) < 4 * np.std(data)])
    print('{} began work'.format(multiprocessing.current_process().name))
    data = np.abs(data)
    data = np.expand_dims(data, axis=0)
    pt = PowerTransformer(method='box-cox')
    pt.fit(data.T)
    print('{} finished'.format(multiprocessing.current_process().name))
    return pt.lambdas_


def main():
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    n_channels = 8
    X = np.empty((n_channels, 0))
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
        if 'Stair' not in file:
            continue
        with open(data_path + '\\' + file) as json_file:
            dict_data = json.load(json_file)
        emg_data = np.array(dict_data["EMG"])
        X = np.hstack((X, emg_data))
    X_std = np.std(X, axis=1)
    X_mean = np.mean(X, axis=1)
    X = (X - X_mean[:, None]) / X_std[:, None]
    a = power_transform(np.expand_dims(np.abs(X[0]),0).T, method='box-cox')
    nProcess = multiprocessing.cpu_count()
    with multiprocessing.Pool(nProcess) as pool:
        lambdas = pool.map(parallel_proc, X)

    with open(data_path + r'\lambdas.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lambdas)


if __name__ == "__main__":
    main()