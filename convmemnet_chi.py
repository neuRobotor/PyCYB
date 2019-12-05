import numpy as np
import sys
import os
import json
import re
import csv
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras import optimizers
from keras.layers import MaxPooling1D
from keras.layers import Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from functools import partial
from scipy.stats import norm, chi
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def conv1d_model(n_timesteps, n_features, n_outputs, drp=0.3, krnl=(3, 3)):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=krnl[0], activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=krnl[1], activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(drp))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, clipnorm=1)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # # fit network
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate model
    # loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return model


def rolling_window(a, window, step):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],) + step
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def kfold():
    # region Input handling
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
    else:
        data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB005\Data'
        # print("Please input directory with data")
        # return
    if len(sys.argv) >= 3:
        freq_factor = sys.argv[2]
    else:
        freq_factor = 20
    if len(sys.argv) >= 4:
        n_channels = int(sys.argv[3])
    else:
        n_channels = 8
    # endregion

    overlap = 2

    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'] if not diff \
        else ['LHipW', 'RHipW', 'LKneeW', 'RKneeW', 'LAnkleW', 'RAnkleW']


    with open(data_path+r'\params.csv') as f:
        reader = csv.reader(f)
        params = list(reader)
        params = [[float(el) for el in row] for row in params]

    X = np.empty((0, overlap*freq_factor, n_channels))
    Y = [[] for _ in range(len(joint_names))]
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
        with open(data_path+'\\'+file) as json_file:
            dict_data = json.load(json_file)
        for i, joint in enumerate(joint_names):
            cur_data = dict_data[joint]
            Y[i].extend(cur_data[(overlap-1):])
        emg_data = np.array(dict_data["EMG"])
        emg_std = np.std(emg_data, axis=1)
        emg_mean = np.mean(emg_data, axis=1)
        emg_data = (emg_data - emg_mean[:, None]) / emg_std[:, None]
        emg_data = np.abs(emg_data)
        for chnl in range(len(emg_data)):
            arg = params[chnl][:-2]
            loc = params[chnl][-2]
            scale = params[chnl][-1]
            a_max = chi.ppf(0.9999999999999999, loc=loc, scale=scale, *arg)
            a_min = chi.ppf(0.000000000001, loc=loc, scale=scale, *arg)
            transf = np.clip(emg_data[chnl, :], a_min=a_min, a_max=a_max)
            transf = chi.cdf(transf, loc=loc, scale=scale, *arg)
            emg_data[chnl] = norm.ppf(transf)
        emg_data = [emg_data[:, i:i + freq_factor] for i in range(0, emg_data.shape[1], freq_factor)]
        emg_data = np.dstack(emg_data)
        emg_data = emg_data.transpose()  # shape: (chunk number, sample, feature)
        emg_data = np.array([np.concatenate([emg_data[i + o, :, :] for o in range(overlap)], axis=0)
                             for i in range(emg_data.shape[0] - (overlap-1))])
        X = np.vstack((X, emg_data))
    Y = np.array(Y)
    Y = Y[:, :, 0].transpose()
    print('Data loaded. Beginning training.')

    # region Neural Network training
    k = 5
    drop = 0.3
    kernel = (5, 3)
    scores = list()
    losses = list()
    cur_model = partial(conv1d_model, n_timesteps=X.shape[1], n_features=X.shape[2],
                        n_outputs=Y.shape[1], drp=drop, krnl=kernel)
    model = cur_model()
    estimator = KerasRegressor(build_fn=cur_model, epochs=200, batch_size=32, verbose=2)
    kfold = KFold(n_splits=k)
    scores = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)

    print(scores)
    m, st = np.mean(scores), np.std(scores)

    print('MSE: {0:.3f} (+/-{1:.3f})'.format(-m, st))
    print('K-fold: {0:.0f}'.format(k))
    # endregion

    # region Self-documentation
    ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
            for f in os.listdir('.') if f.endswith('.txt')]
    if not ends:
        ends = [0]
    with open(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
              str(max(ends) + 1) + '.txt', 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\nMSE: {:.3f}% (+/-{:.3f})\n"
                "Dropout (if applicable): {:.3f}\n"
                "Kernel (if applicable): {:.0f}, {:.0f}\n"
                "K: {:.0f}\n".format(m, st, drop, kernel[0], kernel[1], k)
                + "\n\nUsing Files:\n")
        for file in [os.path.splitext(f)[0] for f in os.listdir(data_path) if f.endswith('.json')]:
            f.write(file + "\n")

    # plot_model(model, to_file='model_plot'+str(max(ends) + 1)+'.png', show_shapes=True, show_layer_names=True)
    # endregion

def main():
    kfold()
    pass


if __name__ == "__main__":
    main()

