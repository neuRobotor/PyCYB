import numpy as np
import sys
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from functools import partial
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 2, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model


def simple_net(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 2, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model


def main():
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

    X = np.empty((0, freq_factor, n_channels))
    Y= np.empty((0, 1))
    print('Loading data...')
    for file in [f for f in os.listdir(data_path) if f.endswith('.csv')]:
        emg_data = np.loadtxt(data_path+'\\'+file, delimiter=",")
        emg_std = np.std(emg_data, axis=1)
        emg_data = emg_data / emg_std[:, None]
        emg_data = [emg_data[:, i:i + freq_factor] for i in range(0, emg_data.shape[1], freq_factor)]
        emg_data = np.dstack(emg_data)
        emg_data = emg_data.transpose()  # shape: (chunk number, sample, feature)
        X = np.vstack((X, emg_data))
        if 'Sit' in file:
            Y = np.vstack((Y, np.ones((emg_data.shape[0], 1))))
        else:
            Y = np.vstack((Y, np.zeros((emg_data.shape[0], 1))))
    Y = to_categorical(Y)
    print('Data loaded. Beginning training.')

    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.4)
    scores = list()
    for r in range(1):
        score, model = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    print(sum(model.predict_classes(X)))


if __name__ == "__main__":
    main()

