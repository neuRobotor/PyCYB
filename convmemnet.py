import numpy as np
import sys
import os
import json
import re
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from functools import partial
from convnet import summary

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def conv1d_model(n_timesteps, n_features, n_outputs, drp=0.3, krnl=(3, 3), dilate=0, mpool=0):
    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=krnl[0], activation='relu', input_shape=(n_timesteps, n_features)))
    if not dilate:
        model.Name = "1D TCN"
        model.add(Conv1D(filters=16, kernel_size=krnl[1], activation='relu'))
    else:
        model.Name = "1D {} dilated TCN".format(dilate)
        model.add(Conv1D(filters=16, kernel_size=krnl[1], activation='relu', dilation_rate=dilate))
    if mpool:
        model.add(MaxPooling1D(pool_size=mpool))
    model.add(Dropout(drp))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def norm_emg(data):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def stack_emg(emg_data, window_size, stride):
    emg_data = [emg_data[:, i:i + window_size] for i in range(0, emg_data.shape[1]-window_size+1, stride)]
    emg_data = np.dstack(emg_data)
    return emg_data.transpose()  # shape: (chunk number, sample, feature)


def data_proc(data_path, method, params=None, freq_factor=20, window_size=20,
              diff=False, n_channels=8, task=None, stride=None):
    if stride is None:
        stride = window_size
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'] if not diff \
        else ['LHipW', 'RHipW', 'LKneeW', 'RKneeW', 'LAnkleW', 'RAnkleW']
    X = np.empty((0, window_size, n_channels))
    Y = [[] for _ in range(len(joint_names))]
    files = list()
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
        if task is not None and task not in file:
            continue
        files.append(file)
        with open(data_path + '\\' + file) as json_file:
            dict_data = json.load(json_file)

        if params is None:
            emg_data = method(np.array(dict_data["EMG"]))
        else:
            emg_data = method(np.array(dict_data["EMG"]), params)
        emg_data = stack_emg(emg_data, window_size=window_size, stride=stride)
        X = np.vstack((X, emg_data))

        for i, joint in enumerate(joint_names):
            cur_data = dict_data[joint]
            upsampled = np.array(cur_data)
            xp = np.arange(len(upsampled))*freq_factor/stride
            x = np.arange(window_size, len(upsampled)*freq_factor+1)
            x = x[0::stride]
            def interp_f(data):
                return np.interp(x, xp, data)
            upsampled = np.apply_along_axis(interp_f, 0, upsampled)
            Y[i].extend(upsampled.tolist())

    Y = np.array(Y)
    Y = Y[:, :, 0].transpose()
    return X, Y, files


def train_net(X, Y, dil, drop, poolsize, kernel, ep, ba, k, validate, window_size, stride, freq_factor):
    cur_model = partial(conv1d_model, n_timesteps=X.shape[1], n_features=X.shape[2],
                       n_outputs=Y.shape[1], drp=drop, krnl=kernel, dilate=dil, mpool=poolsize)
    model = cur_model()

    if not validate:
        file_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data\004_Validation20.json'
        with open(file_path) as json_file:
            dict_data = json.load(json_file)
        emg_data = norm_emg(np.array(dict_data["EMG"]))
        X0 = stack_emg(emg_data, window_size=window_size, stride=stride)
        #X0 = X0[:,:,:-1]
        joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
        Y0 = [[] for _ in range(len(joint_names))]
        for i, joint in enumerate(joint_names):
            cur_data = dict_data[joint]
            upsampled = np.array(cur_data)
            xp = np.arange(len(upsampled))*freq_factor/stride
            x = np.arange(window_size, len(upsampled)*freq_factor+1)
            x = x[0::stride]
            def interp_f(data):
                return np.interp(x, xp, data)
            upsampled = np.apply_along_axis(interp_f, 0, upsampled)
            Y0[i].extend(upsampled.tolist())

        Y0 = np.array(Y0)
        Y0 = Y0[:, :, 0].transpose()
        Y0 = np.expand_dims(Y0[:, 2], 1)

        model.fit(X, Y, batch_size=ba, epochs=ep, verbose=2, callbacks=None, validation_data=(X0, Y0))
        ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
                for f in os.listdir(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Models') if f.endswith('.h5')]
        if not ends:
            ends = [0]
        filepath = r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Models\model_' + str(max(ends) + 1) + '.h5'
        model.save(filepath)
        return

    estimator = KerasRegressor(build_fn=cur_model, epochs=ep, batch_size=ba, verbose=2)
    kfold = KFold(n_splits=k)
    scores = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=3)
    return scores, model


def kfold():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    window_size = 40
    n_channels = 8
    stride = 1
    freq_factor = 20
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]

    X, Y, files = data_proc(data_path, norm_emg, diff=diff,
                            window_size=window_size, n_channels=n_channels, task='Walk', stride=stride)
    #X = X[:, :, :-1]
    Y = np.expand_dims(Y[:, 2], 1)
    print('Data loaded. Beginning training.')

    # ########################
    #       TRAIN NETWORK
    # ########################

    k = 5
    drop = 0.5
    kernel = (3, 3)
    dil = 3
    poolsize = 2
    ep, ba = 50, 100
    validate = False
    if validate:
        scores, model = train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba,
                                  validate=validate, window_size=window_size, stride=stride, freq_factor=freq_factor)
        summary(k, scores, kernel, drop, model, data_path, ep, ba, files)
    else:
        train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba,
                  validate=validate, window_size=window_size, stride=stride, freq_factor=freq_factor)




def main():
    kfold()
    pass


if __name__ == "__main__":
    main()

