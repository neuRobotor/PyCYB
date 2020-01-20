# fit an empirical cdf to a bimodal dataset
from convmemnet_ecdf import ecdf_norm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
import os
import csv
import json
import multiprocessing
import seaborn as sns
import pickle


def parallel_proc(data):
    # data = np.abs(data[abs(data - np.mean(data)) < 4 * np.std(data)])
    print('{} began work'.format(multiprocessing.current_process().name))
    ecdf = ECDF(data)
    print('{} finished'.format(multiprocessing.current_process().name))
    return ecdf


def diff_central(x, y):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    f = (x2 - x1) / (x2 - x0)
    return (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)


def epdf(x, y, N):
    ma_y = np.convolve(y, np.ones((N,)) / N, mode='valid')
    ma_x = np.convolve(x, np.ones((N,)) / N, mode='valid')
    return ma_x[1:-1], diff_central(ma_x, ma_y)


def main():
    sns.set()
    sns.set_context("paper")
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

    X0 = np.array(X)
    X_std = np.std(X, axis=1)
    X_mean = np.mean(X, axis=1)
    X = np.abs((X - X_mean[:, None]) / X_std[:, None])

    nProcess = multiprocessing.cpu_count()
    with multiprocessing.Pool(nProcess) as pool:
        ecdfs = pool.map(parallel_proc, X)

    fig, axes = plt.subplots(4, 2)
    plt.tight_layout()
    for i, ecdf in enumerate(ecdfs):
        x, y = epdf(ecdf.x, ecdf.y, 10)
        axes.flatten()[i].plot(x, y, alpha=0.5, lw=0.4)
        x, y = epdf(ecdf.x, ecdf.y, 1000)
        axes.flatten()[i].plot(x, y)
        axes.flatten()[i].hist(X[i, :], bins=200, density=True, rwidth=1, edgecolor=sns.color_palette()[2])
        axes.flatten()[i].set_yscale("log")
        axes.flatten()[i].set_xlabel("Normalised voltage")
        axes.flatten()[i].set_ylabel("Probability density")

    fig, axes = plt.subplots(4, 2)
    plt.tight_layout()
    for i, ecdf in enumerate(ecdfs):
        axes.flatten()[i].plot(ecdf.x, ecdf.y)
        axes.flatten()[i].set_xlabel("Normalised voltage")
        axes.flatten()[i].set_ylabel("ECDF")

    X1 = ecdf_norm(X0, ecdfs)
    fig, axes = plt.subplots(4, 2)
    plt.tight_layout()
    for i, ecdf in enumerate(ecdfs):
        axes.flatten()[i].hist(X1[i, :], bins=100, density=True, rwidth=1, edgecolor=sns.color_palette()[0])
        if i % 2 is 0:
            axes.flatten()[i].set_ylabel("Probability density")
        if i > 5:
            axes.flatten()[i].set_xlabel("Normalised voltage")
    plt.tight_layout()
    plt.show()

    with open(data_path + r'\ecdfs.pickle', 'wb') as handle:
        pickle.dump(ecdfs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


