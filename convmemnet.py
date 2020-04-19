import numpy as np
import sys
import os
import json
import re

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling1D, MaxPooling2D, DepthwiseConv2D
from tensorflow.keras.layers import Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from functools import partial
from utility.save_load_util import summary
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import normalize
from utility.save_load_util import incr_file, incr_dir, model_summary_to_string, kw_summary, print_to_file

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def depthwise_model(shape_X, shape_Y, drp=0.3, krnl=(3, 3), dilate=0, mpool=0):
    n_timesteps = shape_X[2]
    n_features = shape_X[3]
    n_outputs = shape_Y[1]

    model = Sequential()
    model.add(DepthwiseConv2D(input_shape=(1, n_timesteps, n_features),
                              kernel_size=(1, krnl[0]),
                              depth_multiplier=4,
                              activation='relu',
                              padding='valid'))
    if mpool:
        model.add(MaxPooling2D(pool_size=(1,mpool)))
    if not dilate:
        model.Name = "1D TCN"
        model.add(DepthwiseConv2D(input_shape=(1, n_timesteps, n_features),
                                  kernel_size=(1, krnl[1]),
                                  depth_multiplier=16,
                                  activation='relu',
                                  padding='valid'))
    else:
        model.Name = "1D {} dilated TCN".format(dilate)
        model.add(DepthwiseConv2D(input_shape=(1, n_timesteps, n_features),
                                  kernel_size=(1, krnl[1]),
                                  dilation_rate=dilate,
                                  depth_multiplier=3,
                                  activation='relu',
                                  padding='valid'))
    if mpool:
        model.add(MaxPooling2D(pool_size=(1,mpool)))
    model.add(Dropout(drp))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def conv1d_model(shape_X, shape_Y, drp=0.3, krnl=(3, 3), dilate=0, mpool=0):
    n_timesteps = shape_X[1]
    n_features = shape_X[2]
    n_outputs = shape_Y[1]

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


def stack_emg(emg_data, window_size, stride, windows=None):
    emg_data = [emg_data[:, i:i + window_size] for i in range(0, emg_data.shape[1]-window_size+1, stride)]
    emg_data = np.dstack(emg_data)
    emg_data = emg_data.transpose()
    if windows is None:
        return emg_data  # shape: (chunk number, sample, feature)
    else:
        return [emg_data[:,:windows[0],:], emg_data[:, -windows[1]:, :]]


def data_proc(data_path, method, params=None, freq_factor=20, window_size=20,
              diff=False, n_channels=8, task=None, stride=None, windows=None):
    if stride is None:
        stride = window_size
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'] if not diff \
        else ['LHipW', 'RHipW', 'LKneeW', 'RKneeW', 'LAnkleW', 'RAnkleW']

    if windows is None:
        X = np.empty((0, window_size, n_channels))
    else:
        X = [np.empty((0, w, n_channels)) for w in windows]
    Y = [[] for _ in range(len(joint_names))]
    files = list()
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
        if (task is not None and task not in file) or 'Env' in file:
            continue
        files.append(file)
        with open(data_path + '\\' + file) as json_file:
            dict_data = json.load(json_file)

        if params is None:
            emg_data = method(np.array(dict_data["EMG"]))
        else:
            emg_data = method(np.array(dict_data["EMG"]), params)
        emg_data = stack_emg(emg_data, window_size=window_size, stride=stride, windows=windows)
        if windows is None:
            X = np.vstack((X, emg_data))
        else:
            for i in range(2):
                X[i] = np.vstack((X[i], emg_data[i]))

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


def train_net(X, Y, dil, drop, poolsize, kernel, ep, ba, k, validate, window_size, stride, freq_factor, files):
    cur_model = partial(depthwise_model, shape_X=X.shape, shape_Y=Y.shape, drp=drop, krnl=kernel, dilate=dil, mpool=poolsize)
    model = cur_model()

    if not validate:
        data_path = r'C:\Users\hbkm9\Documents\Projects\CYB\Balint\CYB104\Data'
        X0, Y0, _ = data_proc(data_path, norm_emg,
                            window_size=window_size,  task='Validation', stride=stride, freq_factor=freq_factor)
        Y0 = normalize(np.array(Y0), axis=1)
        #Y0 = np.expand_dims(Y0[:, 2], 1)
        X0 = np.expand_dims(X0, 1)

        mc = ModelCheckpoint('Models/best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
        history = model.fit(X, Y, batch_size=ba, epochs=ep, verbose=2, callbacks=[es, mc], validation_data=(X0, Y0))
        dir_path, ends = incr_dir(r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models', 'model_')
        os.mkdir(dir_path)
        #filepath, ends = incr_file(r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models', 'model_', '.h5')
        model.save(dir_path+'\\model_' + str(max(ends) + 1))
        import pickle
        with open(dir_path+'\\history_' + str(max(ends) + 1) + r'.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        file_names = "\n".join(files)
        str_sum = model_summary_to_string(model) + '\n'
        str_sum += kw_summary(dil=dil, drop=drop, poolsize=poolsize, kernel=kernel,
                              epochs=ep, batch=ba, window_size=window_size, stride=stride)
        str_sum += '\n\n' + file_names
        print_to_file(str_sum, dir_path + '\\summary_'+ str(max(ends) + 1) + r'.txt')
        return




    estimator = KerasRegressor(build_fn=cur_model, epochs=ep, batch_size=ba, verbose=2)
    kfold = KFold(n_splits=k)
    scores = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=3)
    return scores, model


def main():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\hbkm9\Documents\Projects\CYB\Balint\CYB104\Data'
    window_size = 200
    n_channels = 8
    stride = 1
    freq_factor = 20
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]

    X, Y, files = data_proc(data_path, norm_emg, diff=diff,
                            window_size=window_size, n_channels=n_channels, task='Walk', stride=stride)
    #X = X[:, :, :-1]
    #Y = np.expand_dims(Y[:, 2], 1)
    Y = normalize(np.array(Y), axis=1)
    X = np.expand_dims(X, 1)
    print('Data loaded. Beginning training.')

    # ########################
    #       TRAIN NETWORK
    # ########################

    k = 5
    drop = 0.5
    kernel = (15, 3)
    dil = 3
    poolsize = 8
    ep, ba = 50, 20
    validate = False
    if validate:
        scores, model = train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba,
                                  validate=validate, window_size=window_size, stride=stride, freq_factor=freq_factor, files=files)
        summary(k, scores, kernel, drop, model, data_path, ep, ba, files)
    else:
        train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba,
                  validate=validate, window_size=window_size, stride=stride, freq_factor=freq_factor, files=files)


if __name__ == "__main__":
    main()

