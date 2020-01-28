from ECG_filter.anc import lms_ic
from utility.emg_proc import norm_emg
from utility.load_util import load_dict, save_dict, get_file_names
import multiprocessing
import numpy as np


def parallel_proc(filename):
    print('Processing file ' + filename + '...')
    cur_dict = load_dict(filename)
    emg_data = cur_dict["EMG"]
    emg_data = norm_emg(emg_data)
    emg_filt = list()
    for channel in emg_data[:-1]:
        n, x, x_hat, e, ao, F, Ao = lms_ic(3, channel, emg_data[-1], mu=0.01)
        n, x, x_hat, e, ao, F, Ao = lms_ic(3, channel, emg_data[-1], mu=0.01, a0=ao)
        emg_filt.append(e.tolist())
    emg_filt.append(emg_data[-1].tolist())
    cur_dict["EMG"] = emg_filt
    save_dict(filename, cur_dict)


def main():
    file_paths = get_file_names(r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data', task='Walk')
    n_process = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_process) as pool:
        pool.map(parallel_proc, file_paths)
    return


if __name__ == '__main__':
    main()
