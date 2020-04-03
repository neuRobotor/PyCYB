from ECG_filter.lms import *
from ECG_filter.lms_filt import *
from utility.save_load_util import load_emg, load_emg_stack
import matplotlib.pyplot as plt
import scipy.signal as sp
import seaborn as sns
import numpy as np


def norm_emg(data):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def main_delay_sweep():
    # region Loading and setup
    sns.set_style('dark')
    n_delays = 5
    sns.set_palette(sns.cubehelix_palette(n_delays, start=.5, rot=-.75))
    pal = sns.cubehelix_palette(n_delays, start=.5, rot=-.75)
    emg = load_emg_stack(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment_Balint\CYB005\Data', task='Stair')
    # endregion
    filt_ord = 4
    rec = 5
    targ = 6
    n = lambda x: np.tile((x - np.mean(x)) / np.std(x), 2)
    #filters = [anc(n(emg[rec][targ, :]), n(emg[rec][7, :]), 0.00001, rho=1e-08, alg='Benv',
    #               mode='Adap', delay=d, order=filt_ord) for d in range(n_delays)]
    filters = [anc(n(emg[rec][targ, :]), n(emg[rec][7, :]), 0.001, alg='A&F', rho=1e-8,
               mode='GASS', delay=d, order=filt_ord, sign_regr=True) for d in range(n_delays)]
    delay = 1
    ers = list()
    for f in filters:
        # plt.plot(np.arange(delay,delay+len(np.squeeze(f.e))), np.squeeze(f.e).T, zorder=delay*2)
        plt.plot(np.arange(delay, delay + f.W.shape[1]), f.W.T, zorder=delay * 2, color=pal[delay-1])
        delay += 1
        ers.append(np.mean(np.abs(f.e**2)))
    plt.figure()
    plt.plot(np.arange(-filt_ord+1, -filt_ord+1 + len(n(emg[rec][7, :]))), n(emg[rec][7, :]) / 3, color='red', alpha=0.8, zorder=-5)
    plt.plot(np.arange(-filt_ord+1, -filt_ord+1 + len(n(emg[rec][targ, :]))), n(emg[rec][targ, :]), color='orange', alpha=0.8, zorder=-5)
    plt.figure()
    plt.plot(ers, c=pal[int(len(pal)/2)])
    plt.ylabel('MSE'),
    plt.xlabel('delay (samples)')
    plt.title('Left IO, GNGD, sign regressor')
    plt.show()
    return


def crop_to_peaks(emg, ecg, order, delay):
    X = lms_stack(ecg, order)
    X = X[:, :X.shape[1] - delay]
    d = emg[order + delay - 1:]
    peaks, _ = sp.find_peaks(X[0,:], distance=800, height=3.2)
    if min(peaks)<200:
        peaks = peaks[1:]
    if max(peaks)>X.shape[1]-202:
        peaks = peaks[:-1]
    qrs = [X[:, peak_id-200:peak_id +201] for peak_id in peaks]
    segs =[d[peak_id-200:peak_id +201] for peak_id in peaks]
    return np.concatenate(qrs), np.concatenate(segs)

def main_cropped_lms():
    sns.set_style('darkgrid')
    emg = load_emg_stack(r'C:\Users\hbkm9\Documents\Projects\CYB\Balint\CYB104\Data', task='Stand')
    emg = [norm_emg(cur_emg) for cur_emg in emg]
    targ = 5
    cropped = [crop_to_peaks(cur_emg[targ,:], cur_emg[7,:], 4, 0) for cur_emg in emg]
    return


def main_spectrum_lms():
    emg = load_emg(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment_Balint\CYB005\Data\005_Stair12.json')
    emg = norm_emg(emg)
    targ = 1
    plt.plot(emg[targ, :])
    plt.figure()
    F = spectrum_lms(emg[targ, :], 1000, gamma=0.01)
    #spec = norm_emg(np.abs(F.W[1:int(F.W.shape[0]/2), 3:]).T).T
    #spec = spec-np.min(spec)+1
    spec = np.abs(F.W[1:int(F.W.shape[0] / 2), 3:])
    plt.pcolormesh(np.arange(spec.shape[1]), np.arange(spec.shape[0])/spec.shape[0]/2*2000, np.log10(spec))
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    return


def main_spectrum_visu():
    # region Loading and setup
    sns.set_style('darkgrid')
    emg = load_emg(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment_Balint\CYB005\Data\005_Stair12.json', task='Stair')
    emg = norm_emg(emg)
    fig, axes = plt.subplots(2,4)
    names = ['L Internal Oblique','R Internal Oblique','L External Oblique','R External Oblique','L Trapezius', 'R Trapezius', 'Erector Spinae', 'ECG']
    for i in range(8):
        if i is not 7:
            axes.flatten()[i].plot(emg[7,:], alpha=0.6)
        axes.flatten()[i].plot(emg[i,:])
        if i is 0:
            axes.flatten()[i].legend(['ECG', 'EMG' ])
        axes.flatten()[i].set_title(names[i])
    plt.show()
    ecg = emg[7, :]
    emg1 = emg[5, :]
    emg = emg[6, :]

    # endregion
    def plot_stuff():
        # region Spectral plotting
        fig, axs = plt.subplots(1, 3)

        def my_spec(ts_in, ax):
            f, t, Sxx = sp.spectrogram(ts_in, 2000, nperseg=256, noverlap=128, nfft=2000)
            ax.pcolormesh(t, f, np.log(Sxx))
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [sec]')
            ax.set_ylim((0, 500))

        my_spec(emg, axs[0])
        my_spec(emg1, axs[1])
        my_spec(ecg, axs[2])

        # endregion

        # region ECG ARMA modelling

        peaks, _ = sp.find_peaks(ecg, distance=800, height=3.2)
        axs[2].plot(peaks / 2000, np.ones_like(peaks) * 100, 'r+')
        plt.figure()
        plt.plot(ecg)
        qrs = np.zeros((len(peaks) - 1, 260))
        for i in range(len(peaks) - 1):
            qrs[i, :] = ecg[peaks[i] - 59: peaks[i] + 201]
        plt.plot(peaks, np.ones_like(peaks) * 5, 'r+')
        # endregion
        from statsmodels.tsa.stattools import pacf
        pacs = [pacf(cur_ecg, method='ywm') for cur_ecg in qrs]
        plt.figure()
        plt.stem(np.mean(pacs, axis=0), use_line_collection=True)
        plt.axhline(1.96 / np.sqrt(qrs.shape[1]), color='red', ls='--')
        plt.axhline(-1.96 / np.sqrt(qrs.shape[1]), color='red', ls='--')
        plt.show()

    plot_stuff()
    emg1 = np.clip(emg1, -8, 8)
    F = anc(emg1, ecg, 5e-04, rho=1e-08, delay=0, order=4, mode='GASS', alg='A&F', alph=0.9)
    plt.plot(emg1, alpha=0.1)
    plt.plot(ecg / 4, alpha=0.1)
    plt.plot(np.squeeze(F.W).T)
    plt.show()
    return



if __name__ == '__main__':
    main_spectrum_lms()
