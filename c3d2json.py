import sys
import os
from jcs import Segment, JCS
from functools import partial
import multiprocessing


def proc_input(list_in):
    list_out = list(list_in[1:])
    options_out = list()
    for el in list_in:
        if "-" in el:
            options_out.append(el)
            list_out.remove(el)
    return list_out, options_out


def c3d_proc(file_path, options):

    return


def multi_worker(file_path, target_dir, options):

    return


def dir_proc(paths, options):

    for path in paths:
        if not os.path.isdir(dir):
            print("No such directory found:")
            print(dir)
            help_msg()
            return

    dir_in = paths[0]
    target_dir = paths[0]
    if len(paths) > 1:
        target_dir = paths[1]

    f = partial(multi_worker, dir_in, target_dir, options)
    fnames = [dir_in + '\\' + f for f in os.listdir(dir_in) if f.endswith('.c3d')]

    n_process = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_process) as pool:
        pool.map(f, fnames)
    return


def help_msg():
    print('Joint Angle Estimation using Joint-Coordinate Systems and EMG pre-processing\n'
          'Usage:\n'
          'python c3dcyb.py [options] source_path [target_path]\n'
          '\t[options]:\n'
          '\t\tnone\t\tInput source_path to directory of .c3d files to process\n'
          '\t\t-d --diff\tExtract joint angular velocity instead of angles\n'
          '\t\t-cfN --cutoffN\tSubstitute N with cutoff frequency (Hz) for EMG low-pass\n'
          '\t\t-e --enve\tCalculate envelope of EMG using supplied cf\n'
          '\t\t\t\tOptionally specify target_path if target directory is different\n'
          '\t\t-h --help\tDisplay this message\n'
          '\t\t-v --visu\tProcess a single .c3d file, and display interactive plots\n'
          '\t\t-z --zdir\tIndicate vertical travel in name (Up or Down)\n'
          '\t\t-p --para\tExtract gait parameters instead of angles\n'
          '\t\t\t\tNo target_path needed\n')

def main():
    if len(proc_input(sys.argv)[0]) > 0:
        if '-h' in sys.argv or '--help' in sys.argv:
            help_msg()
            return
        #TODO
        # if '-v' in sys.argv or '--visu' in sys.argv:
        #     visu(diff='-d' in sys.argv or '--diff' in sys.argv, env='-e' in sys.argv or '--enve' in sys.argv)
        #     return
        else:
            dir_proc(*proc_input(sys.argv))
    else:
        print('Please use at least 1 path argument\n')
        help_msg()


if __name__ == "__main__":
    main()