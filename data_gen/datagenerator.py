import numpy as np
import keras
import json
from tensorflow.keras.utils import Sequence, to_categorical


def _raise(ex):
    raise ex


def _pass(x, **kwargs):
    return x


class TCNDataGenerator(Sequence):
    # Generates data for Keras
    def __init__(self, data_dir, file_names, window_size=32, stride=1, batch_size=32, freq_factor=20, delay=1,
                 dims=(0,), shuffle=True, joint_names=('LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'),
                 preproc=_pass, angproc=_pass, ppkwargs=None, gap_windows=None, channel_mask=None, time_step=1):
        'Initialization'

        self.batch_size = batch_size
        self.delay = delay
        self.data_dir = data_dir
        self.dims = dims  # Spacial dims of angles
        self.shuffle = shuffle

        self.file_names = sorted([f for f in file_names if f.endswith('.json')])

        self.window_size = window_size
        self.stride = stride
        self.joint_names = joint_names
        self.freq_factor = freq_factor
        self.gap_windows = gap_windows
        self.channel_mask = channel_mask
        self.time_step = time_step

        self.preproc = preproc
        self.angproc = angproc
        if ppkwargs is None:
            self.ppkwargs = {}
        else:
            self.ppkwargs = ppkwargs

        self.n_windows = 0
        self.window_index_heads = list()
        self.emg_data = list()
        self.angle_data = list()
        self.load_files()
        if self.emg_data:
            self.n_channels = len(self.emg_data[0])
        if self.angle_data:
            self.n_angles = len(self.angle_data[0])
        self.window_idx = np.arange(self.n_windows)
        self.on_epoch_end()

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        self.window_index_heads = list()
        for file in self.file_names:
            with open(self.data_dir + '\\' + file) as json_file:
                dict_data = json.load(json_file)
                if self.channel_mask is None:
                    self.emg_data.append(self.preproc(np.array(dict_data["EMG"]), **self.ppkwargs))
                else:
                    self.emg_data.append(
                        self.preproc(np.array([dict_data["EMG"][i] for i in self.channel_mask]), **self.ppkwargs))
                self.window_index_heads.append((self.n_windows,
                                                self.n_windows +
                                                int((len(self.emg_data[-1][0]) - self.window_size + 1) / self.stride)))
                self.n_windows = self.window_index_heads[-1][1]
                angles = list()
                for joint in self.joint_names:
                    xp = np.arange(len(dict_data[joint])) * self.freq_factor
                    x = np.arange(0, len(dict_data[joint]) * self.freq_factor + self.delay)

                    def interp_f(data):
                        return np.interp(x, xp, data)

                    upsampled = np.apply_along_axis(interp_f, 0, dict_data[joint])
                    angles.append(self.angproc(upsampled))
                self.angle_data.append(np.array(angles))
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_windows / self.batch_size))

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        # Generate indexes of the batch
        cur_indexes = self.window_idx[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        # Generate data
        X, Y = self.data_generation(cur_indexes)
        return X, Y

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]

        X = np.array(
            [self.emg_data[file_id][:, win_id * self.stride:
                                       win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])
        Y = np.array(
            [np.squeeze(self.angle_data[file_id][:, win_id * self.stride + self.delay, list(self.dims)])
             for file_id, win_id in ids])
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.window_idx = np.arange(self.n_windows)
        if self.shuffle:
            np.random.shuffle(self.window_idx)

    def save(self, path, unload=True):
        import pickle
        if unload:
            self.emg_data = list()
            self.angle_data = list()
            self.n_windows = 0
            self.window_index_heads = list()
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class EDTCNGenerator(TCNDataGenerator):

    def __init__(self, decode_length=400, **kwargs):
        self.decode_length = decode_length
        super(EDTCNGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        for file in self.file_names:
            with open(self.data_dir + '\\' + file) as json_file:
                dict_data = json.load(json_file)
                if self.channel_mask is None:
                    self.emg_data.append(self.preproc(np.array(dict_data["EMG"]), **self.ppkwargs))
                else:
                    self.emg_data.append(
                        self.preproc(np.array([dict_data["EMG"][i] for i in self.channel_mask]), **self.ppkwargs))
                self.window_index_heads.append((self.n_windows,
                                                self.n_windows +
                                                int((len(self.emg_data[-1][0]) - self.window_size + 1) / self.stride)))
                self.n_windows = self.window_index_heads[-1][1]
                angles = list()
                for joint in self.joint_names:
                    xp = np.arange(len(dict_data[joint])) * self.freq_factor
                    x = np.arange(0, len(dict_data[joint]) * self.freq_factor + self.delay+self.decode_length)

                    def interp_f(data):
                        return np.interp(x, xp, data)

                    upsampled = np.apply_along_axis(interp_f, 0, dict_data[joint])
                    angles.append(self.angproc(upsampled))
                self.angle_data.append(np.array(angles))
        return

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]

        X = np.array(
            [self.emg_data[file_id][:, win_id * self.stride:
                                       win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])
        Y = np.array(
            [np.squeeze(self.angle_data[file_id][:, win_id * self.stride + self.delay:
                                                    win_id * self.stride + self.delay + self.decode_length:
                                                    self.stride,
                                                    list(self.dims)].flatten())
             for file_id, win_id in ids])
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y


class TCNClassGenerator(TCNDataGenerator):
    def __init__(self, class_enum=('Walk', 'Sit', 'Stair'), **kwargs):
        self.class_enum = class_enum
        super(TCNClassGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        for file in self.file_names:
            with open(self.data_dir + '\\' + file) as json_file:
                dict_data = json.load(json_file)
                if self.channel_mask is None:
                    self.emg_data.append(self.preproc(np.array(dict_data["EMG"]), **self.ppkwargs))
                else:
                    self.emg_data.append(
                        self.preproc(np.array([dict_data["EMG"][i] for i in self.channel_mask]), **self.ppkwargs))
                self.window_index_heads.append((self.n_windows,
                                                self.n_windows +
                                                int((len(self.emg_data[-1][0]) - self.window_size + 1) / self.stride)))
                self.n_windows = self.window_index_heads[-1][1]
        return

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]

        X = np.array(
            [self.emg_data[file_id][:,
             win_id * self.stride:win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])
        Y = to_categorical([[i for i, e in enumerate(self.class_enum) if e in self.file_names[file_id]][0]
                            for file_id, _ in ids], num_classes=len(self.class_enum))
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y
