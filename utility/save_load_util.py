import json
import numpy as np
import os
import re
import sys


def summary(k, scores, kernel, drop, model, data_path, epochs, batch, files):
    print(scores)
    m, st = np.mean(scores), np.std(scores)

    print('MSE: {0:.3f} (+/-{1:.3f})'.format(-m, st))
    print('K-fold: {0:.0f}'.format(k))

    # region Self-documentation

    file_name = incr_file(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries', r'model_summary', '.txt')

    ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
            for f in os.listdir(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries') if f.endswith('.txt')]
    if not ends:
        ends = [0]
    print(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
          str(max(ends) + 1) + '.txt')
    from datetime import datetime
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(r'C:\Users\win10\Desktop\Projects\CYB\PyCYB\Summaries\model_summary' +
              str(max(ends) + 1) + '.txt', 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\nDate: {}\n"
                "File: {}\n"
                "Scores: {}\n"
                "MSE: {:.3f} (+/-{:.3f})\n"
                "Dropout (if applicable): {:.3f}\n"
                "Kernel (if applicable): {:.0f}, {:.0f}\n"
                "Epochs: {}, Batch size: {}\n"
                "K: {:.0f}\n".format(dt_string, os.path.basename(sys.argv[0]),
                                     scores, m, st, drop, kernel[0], kernel[1], epochs, batch, k)
                + "\n\nUsing Files:\n")
        for file in files:
            f.write(file + "\n")


def incr_file(dir_path, file_name, ext):
    ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
            for f in os.listdir(dir_path) if f.endswith(ext)
            and file_name in f]
    if not ends:
        ends = [0]
    return dir_path + '\\', file_name + str(max(ends) + 1) + ext, ends


def get_file_names(dir_path, task=None):
    return [dir_path + '\\' + file for file in sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
            if task is None or task in file]


def load_dict(file_path):
    with open(file_path) as json_file:
        dict_data = json.load(json_file)
    return dict_data


def save_dict(file_path, dict_in):
    with open(file_path, 'w') as fp:
        json.dump(dict_in, fp, indent=4)
    return


def load_emg_stack(path, task='None', n_channels=8):
    emg_stack = list()
    for file in sorted([f for f in os.listdir(path) if f.endswith('.json') and (task in f or task is 'None')]):
        with open(path + '\\' + file) as json_file:
            dict_data = json.load(json_file)
            emg_stack.append(np.array(dict_data["EMG"]))
    return emg_stack


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
