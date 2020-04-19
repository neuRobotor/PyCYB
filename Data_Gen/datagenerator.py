import numpy as np
import keras
import json
from tensorflow.keras.utils import Sequence


def _raise(ex):
    raise ex


class TCNDataGenerator(Sequence):
    # Generates data for Keras
    def __init__(self, data_dir, file_names, window_size=32, stride=1, batch_size=32, freq_factor=20, delay=1,
                 dims=(0,), shuffle=True, joint_names=('LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'),
                 preproc=None):
        'Initialization'

        self.batch_size = batch_size
        self.delay = delay
        self.data_dir = data_dir
        self.dims = dims

        self.shuffle = shuffle
        self.file_names = sorted([f for f in file_names if f.endswith('.json')])
        self.indexes = list()
        self.n_windows = 0
        self.window_size = window_size
        self.stride = stride
        self.joint_names = joint_names
        self.preproc = preproc
        self.freq_factor = freq_factor

        self.emg_data = list()
        self.angle_data = list()
        self.load_files()
        self.n_channels = len(self.emg_data[0])
        self.n_angles = len(self.angle_data[0])
        self.window_idx = np.arange(self.n_windows)
        self.on_epoch_end()

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        for file in self.file_names:
            with open(self.data_dir + '\\' + file) as json_file:
                dict_data = json.load(json_file)
                self.indexes.append(list(self.n_windows +
                                         np.arange(0, int(
                                             (len(dict_data["EMG"][0]) - self.window_size + 1) / self.stride))))
                self.n_windows += len(self.indexes[-1])
                self.emg_data.append(self.preproc(np.array(dict_data["EMG"])))
                angles = list()
                for joint in self.joint_names:
                    xp = np.arange(len(dict_data[joint])) * self.freq_factor / self.stride
                    x = np.arange(0, len(dict_data[joint]) * self.freq_factor + self.delay, self.stride)

                    def interp_f(data):
                        return np.interp(x, xp, data)

                    upsampled = np.apply_along_axis(interp_f, 0, dict_data[joint])
                    angles.append(upsampled)
                self.angle_data.append(np.array(angles))
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_windows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        cur_indexes = self.window_idx[index * self.batch_size:min(len(self.window_idx), (index + 1) * self.batch_size)]

        # Generate data
        X, y = self.data_generation(cur_indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.window_idx = np.arange(self.n_windows)
        if self.shuffle:
            np.random.shuffle(self.window_idx)

    def data_generation(self, cur_indexes):
        X = np.empty((self.batch_size, self.window_size, self.n_channels))
        Y = np.empty((self.batch_size, self.n_angles))

        for i, idx in enumerate(cur_indexes):
            file_id, win_id = [(file_id, wins.index(idx))
                               for file_id, wins in enumerate(self.indexes)
                               if idx in wins][0]

            X[i] = self.emg_data[file_id][:, win_id * self.stride:win_id * self.stride + self.window_size]
            Y[i] = self.angle_data[file_id][win_id * self.stride + self.delay][list(self.dims)]
        return self.gen_method
