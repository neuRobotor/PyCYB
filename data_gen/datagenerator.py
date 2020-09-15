import numpy as np
import json
from tensorflow.keras.utils import Sequence, to_categorical
import copy
import os


def _raise(ex):
    raise ex


def _pass(x, **kwargs):
    return x


class TCNDataGenerator(Sequence):
    # Generates data for Keras
    def __init__(self, data_dir, file_names, window_size=32, stride=1, batch_size=32, freq_factor=20, delay=1,
                 dims=(0,), shuffle=True, joint_names=('LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'),
                 preproc=_pass, angproc=_pass, ppkwargs=None, gap_windows=None, channel_mask=None, time_step=1):

        self.k_idx = list()

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
            with open(os.path.join(self.data_dir, file)) as json_file:
                dict_data = json.load(json_file)
                if self.channel_mask is None:
                    if type(dict_data["EMG"]) is dict:
                        self.emg_data.append(self.preproc(np.array(list(dict_data["EMG"].values())), **self.ppkwargs))
                    else:
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

                    upsampled = np.apply_along_axis(interp_f, 0, self.angproc(dict_data[joint]))
                    angles.append(upsampled)
                self.angle_data.append(np.array(angles))
        self.on_epoch_end()
        return

    def force_unwrap(self):  # only for first angle for now!
        angle_means = list()
        for i in range(self.n_angles):
            angle_means.append(np.mean(self.angle_data[0][i][::self.freq_factor, 0]))

        for f in range(1, len(self.angle_data)):
            for a in range(self.n_angles):
                self.angle_data[f][a][:, 0] = np.unwrap(self.angle_data[f][a][:, 0]*4)/4
                if np.mean(np.mean(self.angle_data[f][a][::self.freq_factor, 0])) < angle_means[a] - 3:
                    self.angle_data[f][a][:, 0] = self.angle_data[f][a][:, 0] + np.pi
                elif np.mean(np.mean(self.angle_data[f][a][::self.freq_factor, 0])) > angle_means[a] + 3:
                    self.angle_data[f][a][:, 0] = self.angle_data[f][a][:, 0] - np.pi

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.window_idx) / self.batch_size))

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
        if not self.k_idx:
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
            self.window_idx = np.array([])
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def validation_split(self, k=2, file_split=True, file_shuffle=True):
        if not file_split:
            self.k_idx = np.array_split(self.window_idx, k)
        else:
            file_idx = np.array(self.window_index_heads)
            if file_shuffle:
                np.random.shuffle(file_idx)
            self.k_idx = np.array_split(file_idx, k)
            for i in range(len(self.k_idx)):
                buff = np.array([])
                for j in range(len(self.k_idx[i])):
                    buff = np.append(buff, np.arange(*self.k_idx[i][j]))
                self.k_idx[i] = buff.astype(int)

    def get_k(self, cur_k, k=2, **kwargs):
        if not self.k_idx:
            self.validation_split(k, **kwargs)

        ks = list(self.k_idx)
        val_idx = ks.pop(cur_k)
        train_idx = np.concatenate(ks)
        valid_gen = copy.copy(self)
        valid_gen.window_idx = val_idx
        self.window_idx = train_idx
        self.on_epoch_end()
        valid_gen.on_epoch_end()
        return valid_gen

    def show(self):
        import matplotlib.pyplot as plt
        _, y0 = self.data_generation(np.sort(self.window_idx))
        plt.plot(y0)
        plt.show()


class EDTCNGenerator(TCNDataGenerator):

    def __init__(self, decode_length=400, **kwargs):
        self.decode_length = decode_length
        super(EDTCNGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file)) as json_file:
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
            with open(os.path.join(self.data_dir, file)) as json_file:
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


class StepTCNGenerator(TCNDataGenerator):
    def __init__(self, **kwargs):
        self.step_classes = list()
        super(StepTCNGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file)) as json_file:
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

                step_class = np.repeat(dict_data["step_class"], self.freq_factor)
                step_class = np.append(step_class, [step_class[-1]]*max(self.delay-self.window_size+1, 0))
                self.step_classes.append(step_class)
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

        Y = to_categorical(
            [self.step_classes[file_id][win_id * self.stride + self.delay] for file_id, win_id in ids], num_classes=3)
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y


class ParamTCNGenerator(TCNDataGenerator):
    def __init__(self, params=("step_heights", "stride_lengths", "step_speed"), **kwargs):
        self.step_params = list()
        self.params = params
        super(ParamTCNGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file)) as json_file:
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

                cur_params = list()
                for p in self.params:
                    cur_value = np.array(dict_data[p])
                    cur_value[:] = 1
                    cur_value[np.array(dict_data[p]) == 0] = 0
                    cur_value[np.array(dict_data[p]) > np.abs(np.max(dict_data[p]))*0.8] = 2
                    cur_params.append(np.atleast_2d(np.repeat(cur_value, self.freq_factor)))

                step_param = np.vstack(cur_params)
                step_param = np.hstack((step_param,
                                        np.tile(np.atleast_2d(step_param[:, -1]).T,
                                                (1, max(self.delay-self.window_size+1, 0)))))
                self.step_params.append(self.angproc(step_param))
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
        Y = to_categorical(
            [self.step_params[file_id][:, win_id * self.stride + self.delay] for file_id, win_id in ids], num_classes=3)
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y