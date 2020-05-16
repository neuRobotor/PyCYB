import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from data_gen.preproc import bp_filter, norm_emg


def plot_history(history, data_dir):
    sns.set()
    sns.set_context('paper')
    plt.figure()
    f, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].legend(('MSE', 'Val MSE'), loc='upper right')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('MSE')

    axs[1].plot(history.history['mape'])
    axs[1].plot(history.history['val_mape'])
    axs[1].legend(('Mape', 'Val Mape'), loc='upper right')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAPE')

    fig = plt.gcf()
    fig.set_size_inches((15.6, 8.775), forward=False)
    plt.savefig(data_dir + '//' + "history_"+ "{:.4f}".format(np.min(history.history["val_loss"])) + ".svg")
    return


def plot_pred(emg_data, Y0, ecg, delay, model, data_dir):
    sns.set()
    sns.set_context('paper')
    plt.figure()
    y = model.predict(emg_data)
    N = 20

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


def main():
    sns.set()
    sns.set_context('paper')

    model_num = 62
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

    gen.file_names = ['004_Validation20.json']
    gen.load_files()
    emg_data, Y0 = gen.data_generation(range(gen.window_index_heads[-1][-1]))
    ecg = gen.emg_data[0][7]
    y = model.predict(emg_data)
    N = 20

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
    # fig.suptitle("Prediction with separated channels", x=0.5, size=22)
    plt.show()


if __name__ == '__main__':
    main()




