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
from utility.save_load_util import summary
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def eval_model(model, trainX, trainy, testX, testy, epochs=200, batch_size=5, verbose=32):
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model, loss

def conv1d_model(n_timesteps, n_features, n_outputs, drp=0.3, krnl=(3, 3)):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=krnl[0], activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=krnl[1], activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(drp))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # # fit network
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate model
    # loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return model


def conv2d_model(trainX, trainy, testX, testy, drp=0.3, krnl=(3, 3)):
    verbose, epochs, batch_size = 2, 200, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Reshape((n_timesteps, n_features, 1)))
    model.add(Conv2D(filters=128, kernel_size=(krnl[0], 8), activation='relu', input_shape=(n_timesteps, n_features, 1)))
    model.add(Reshape((18, 128)))
    model.add(Conv1D(filters=64, kernel_size=krnl[1], activation='relu'))
    model.add(Dropout(drp))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model, loss


def simple_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 2, 200, 32
    # create model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(100, input_dim=160, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, input_dim=100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, input_dim=50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, input_dim=50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, input_dim=50, kernel_initializer='normal', activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model, loss


def kfold():
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

    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'] if not diff \
        else ['LHipW', 'RHipW', 'LKneeW', 'RKneeW', 'LAnkleW', 'RAnkleW']
    X = np.empty((0, freq_factor, n_channels))
    Y = [[] for _ in range(len(joint_names))]
    files=list()
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.json')]):
        files.append(file)
        with open(data_path+'\\'+file) as json_file:
            dict_data = json.load(json_file)
        for i, joint in enumerate(joint_names):
            cur_data = dict_data[joint]
            Y[i].extend(cur_data)
        emg_data = np.array(dict_data["EMG"])
        emg_std = np.std(emg_data, axis=1)
        emg_mean = np.mean(emg_data, axis=1)
        emg_data = (emg_data - emg_mean[:, None]) / emg_std[:, None]
        emg_data = [emg_data[:, i:i + freq_factor] for i in range(0, emg_data.shape[1], freq_factor)]
        emg_data = np.dstack(emg_data)
        emg_data = emg_data.transpose()  # shape: (chunk number, sample, feature)
        X = np.vstack((X, emg_data))
    Y = np.array(Y)
    Y = Y[:, :, 0].transpose()
    print('Data loaded. Beginning training.')
    k = 5
    drop = 0.3
    kernel = (3, 3)
    scores = list()
    losses = list()
    cur_model = partial(conv1d_model, n_timesteps=X.shape[1], n_features=X.shape[2],
                        n_outputs=Y.shape[1], drp=drop, krnl=kernel)
    model = cur_model()
    ep, ba = 400, 32
    estimator = KerasRegressor(build_fn=cur_model, epochs=ep, batch_size=ba, verbose=2)
    kfold = KFold(n_splits=k)
    scores = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
    #
    # for r in range(1):
    #     score, model,loss = conv1d_model(trainX, trainy, testX, testy, drop, kernel)
    #     score = score * 100.0
    #     print('>#%d: %.3f' % (r + 1, score))
    #     scores.append(score)
    #     losses.append(loss)
    # summarize results
    summary(k, scores, kernel, drop, model, data_path, ep, ba, files)

    # plot_model(model, to_file='model_plot'+str(max(ends) + 1)+'.png', show_shapes=True, show_layer_names=True)
    # endregion

def main():
    kfold()
    pass


if __name__ == "__main__":
    main()

