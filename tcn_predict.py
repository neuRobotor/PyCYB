import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from scipy.signal import medfilt, savgol_filter
from data_gen.preproc import bp_filter, norm_emg, spec_proc
from utility.save_load_util import load_emg_stack
import os


def plot_history(history, data_dir):
    sns.set()
    sns.set_context('paper')
    plt.figure()
    f, axs = plt.subplots(nrows=1, ncols=int(len(history)/2))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    legends = [[] for _ in range(int(len(history)/2))]
    for i, (k, v) in enumerate(history.items()):
        axs[i%int(len(history)/2)].plot(v)
        legends[i%int(len(history)/2)].append(k)

    for i in range(int(len(history)/2)):
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(legends[i][0])
        axs[i].legend(legends[i])

    fig = plt.gcf()
    fig.set_size_inches((15.6, 6.775), forward=False)
    plt.savefig(data_dir + '//' + "history_"+ "{:.4f}".format(np.min(history["val_loss"])) + ".svg")
    return


def plot_pred(emg_data, Y0, ecg, delay, model, data_dir, stride):
    sns.set()
    sns.set_context('paper')
    plt.figure()
    y = model.predict(emg_data)
    N = max(1, 20)

    def f(y_in):
        return np.convolve(y_in, np.ones((N,)) / N, mode='valid')

    y = np.apply_along_axis(f, 0, y)
    fig, axes = plt.subplots(3, 2)
    axes = axes.flatten()
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    for i, joint in enumerate(joint_names):
        e, = axes[i].plot(np.arange(len(ecg)) / 2000,
                          (ecg - np.mean(ecg)) / np.std(ecg) / 5 * np.std(Y0[:, i]) + np.mean(Y0[:, i]) + 0.2,
                          alpha=0.25, color='gray', lw=0.7)
        o, = axes[i].plot((np.arange(len(Y0[:, i])) + delay) / 2000, Y0[:, i])
        est, = axes[i].plot((np.arange(N + delay, len(y[:, i]) + N + delay)) / 2000, y[:, i])
        if i > 3:
            axes[i].set_xlabel("Time (s)")
        else:
            axes[i].set_xticklabels([])
        if i == 1:
            plt.legend((o, est, e), ("Actual Angles", "Predicted Angles", "ECG signal (A.U.)"),
                       bbox_to_anchor=(1.04, 0.5),
                       loc="center left", borderaxespad=0)
        axes[i].set_ylabel("Radians")
        axes[i].set_title(joint)
        axes[i].locator_params(axis='y', nbins=4)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    fig = plt.gcf()
    fig.set_size_inches((15.6, 8.775), forward=False)
    plt.savefig(data_dir + '//' + "val_plot.svg")

    return


def plot_now(emg_data, Y0, ecg, delay, model, data_dir):
    sns.set()
    sns.set_context('paper')
    plt.figure()
    y = model.predict(emg_data)
    N = max(1, 20)

    def f(y_in):
        return np.convolve(y_in, np.ones((N,)) / N, mode='valid')

    y = np.apply_along_axis(f, 0, y)
    fig, axes = plt.subplots(3, 2)
    axes = axes.flatten()
    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    for i, joint in enumerate(joint_names):
        e, = axes[i].plot(np.arange(len(ecg)) / 2000,
                          (ecg - np.mean(ecg)) / np.std(ecg) / 5 * np.std(Y0[:, i]) + np.mean(Y0[:, i]) + 0.2,
                          alpha=0.25, color='gray', lw=0.7)
        o, = axes[i].plot((np.arange(len(Y0[:, i])) + delay) / 2000, Y0[:, i])
        est, = axes[i].plot((np.arange(N + delay, len(y[:, i]) + N + delay)) / 2000, y[:, i])
        if i > 3:
            axes[i].set_xlabel("Time (s)")
        else:
            axes[i].set_xticklabels([])
        if i == 1:
            plt.legend((o, est, e), ("Actual Angles", "Predicted Angles", "ECG signal (A.U.)"),
                       bbox_to_anchor=(1.04, 0.5),
                       loc="center left", borderaxespad=0)
        axes[i].set_ylabel("Radians")
        axes[i].set_title(joint)
        axes[i].locator_params(axis='y', nbins=4)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()

    return


def main():
    sns.set()
    sns.set_context('paper')

    model_num = 239
    model = load_model('Models/model_' + str(model_num) + '/best_model_' + str(model_num) + '.h5')

    with open('Models/model_' + str(model_num) + '/gen_' + str(model_num) + '.pickle', "rb") as input_file:
        gen = pickle.load(input_file)

    gen.emg_data = list()
    gen.angle_data = list()
    gen.n_windows = 0
    gen.window_index_heads = list()

    ws = gen.window_size
    s = gen.stride
    delay = gen.delay

    gen.data_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Validation'
    gen.file_names = sorted([f for f in os.listdir(gen.data_dir) if f.endswith('.json') and "Walk" in f])
    gen.stride = 1
    gen.load_files()

    cur_indexes = range(gen.window_index_heads[-1][-1])
    head_tails = gen.window_index_heads
    ids = [(file_id, cur_idx - head_tails[file_id][0])
           for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
           if head_tail[0] <= cur_idx < head_tail[1]]

    Y0 = np.array(
        [np.squeeze(gen.angle_data[file_id][:, win_id + int(gen.delay / gen.stride), gen.dims])
         for file_id, win_id in ids])
    gen.window_idx = np.arange(gen.n_windows)
    raw = load_emg_stack(gen.data_dir, task='Walk', n_channels=8)
    ecg = raw[0][7]
    y = model.predict(gen)
    N = 20

    def f(y_in):
        buff = medfilt(y_in, 51)
        #return np.convolve(buff, np.ones((N,)) / N, mode='valid')
        return

    y = np.apply_along_axis(f, 0, y)
    fig, axes = plt.subplots(3, 2)
    axes = axes.flatten()

    joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    for i, joint in enumerate(joint_names):
        e, = axes[i].plot(np.arange(len(ecg)) / 2000,
                          (ecg - np.mean(ecg)) / np.std(ecg) / 5 * np.std(Y0[:, i]) + np.mean(Y0[:, i]) + 0.2,
                          alpha=0.25, color='gray', lw=0.7)
        o, = axes[i].plot((np.arange(len(Y0[:, i])) + delay) / 2000, Y0[:, i])
        est, = axes[i].plot((np.arange(N + delay, len(y[:, i]) + N + delay)) / 2000, y[:, i])
        if i > 3:
            axes[i].set_xlabel("Time (s)")
        else:
            axes[i].set_xticklabels([])
        if i == 1:
            plt.legend((o, est, e), ("Actual Angles", "Predicted Angles", "ECG Signal (A.U.)"),
                       bbox_to_anchor=(1.04, 0.5),
                       loc="center left", borderaxespad=0)
        axes[i].set_ylabel("Radians")
        axes[i].set_title(joint)
        axes[i].locator_params(axis='y', nbins=4)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    # fig.suptitle("Prediction with separated channels", x=0.5, size=22)
    plt.show()


if __name__ == '__main__':
    main()




