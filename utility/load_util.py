import json
import numpy as np
import os

def load_dict(file_path):
    with open(file_path) as json_file:
        dict_data = json.load(json_file)
    return dict_data


def load_emg(path, task=None, n_channels=8):
    X = np.empty((n_channels, 0))
    if os.path.isdir(path):
        for file in sorted([f for f in os.listdir(path) if f.endswith('.json')]):
            if task not in file and task is not None:
                continue
            with open(path + '\\' + file) as json_file:
                dict_data = json.load(json_file)
                X = np.concatenate((X, dict_data["EMG"]), axis=1)
        return X
    with open(path) as json_file:
        dict_data = json.load(json_file)
    return np.array(dict_data["EMG"])