import multiprocessing
import numpy as np
from ECG_filter.anc import lms_ic
from utility.load_util import load_emg
import pickle


# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
# import ECG_filter.anc

def norm_emg(data):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def parallel_proc(data):
    s, y = data
    print('{} began work'.format(multiprocessing.current_process().name))
    n, x, x_hat, e, ao, F, Ao = lms_ic(3, s, y, mu=0.01)
    print('{} finished'.format(multiprocessing.current_process().name))
    return ao


def main():
    data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data'
    file_path = r'\004_Walk03.json'
    s = load_emg(data_path+file_path, task='Walk')
    s = norm_emg(s)
    data = list()
    for i in range(7):
        data.append((s[i, :], s[7, :]))

    nProcess = min(multiprocessing.cpu_count(), 7)
    with multiprocessing.Pool(nProcess) as pool:
        aos = pool.map(parallel_proc, data)

    with open(data_path + r'\aos.pickle', 'wb') as handle:
        pickle.dump(aos, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == "__main__":
    main()

