import multiprocessing
import numpy as np

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import ECG_filter.anc


def parallel_proc(data):
    # data = np.abs(data[abs(data - np.mean(data)) < 4 * np.std(data)])
    print('{} began work'.format(multiprocessing.current_process().name))
    ecdf = ECDF(data)
    print('{} finished'.format(multiprocessing.current_process().name))
    return ecdf


def main():

    return


if __name__ == "__main__":
    main()

