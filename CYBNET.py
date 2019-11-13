import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import os
import json
from keras import models
from keras import layers
from keras.layers import convolutional as conv
from functools import partial
'''
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# load dataset
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define base model

# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''

def baseline_model(n_timesteps, n_features, n_outputs):
    model = models.Sequential()
    model.add(conv.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(conv.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

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
    for file in [f for f in os.listdir(data_path) if f.endswith('.csv')]:
        emg_data = np.loadtxt(data_path+'\\'+file, delimiter=",")
        emg_data = [emg_data[:, i:i + freq_factor] for i in range(0, emg_data.shape[1], freq_factor)]
        emg_data = np.dstack(emg_data)
        emg_data = emg_data.transpose()  # shape: (chunk number, sample, feature)
        X = np.vstack((X, emg_data))

    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    Y = np.empty(X.shape[0])
    for file in [f for f in os.listdir(data_path) if f.endswith('.json')]:
        with open(data_path+'\\'+file) as json_file:
            dict_joint_angles = json.load(json_file)
            list_vals = [np.array(li) for li in dict_joint_angles.values()]
            keys = dict_joint_angles.keys()
            dict_joint_angles = {key: ar for key, ar in zip(keys, list_vals)}
        cur_Y = np.empty(dict_joint_angles[joint_names[0]].shape[0])
        for joint in joint_names:
            cur_Y = np.vstack((cur_Y, dict_joint_angles[joint][:, 0]))
        cur_Y = cur_Y.transpose()
        Y = np.vstack((Y, cur_Y))

    cur_model = partial(baseline_model, 0)


if __name__ == "__main__":
    main()

