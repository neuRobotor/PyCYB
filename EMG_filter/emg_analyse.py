from EMG_filter.lms import *
from EMG_filter.lms_filt import *
from utility.save_load_util import *
import matplotlib.pyplot as plt
import scipy.signal as sp
import seaborn as sns
import numpy as np
from data_gen.preproc import norm_emg
from matplotlib.gridspec import GridSpec
from matplotlib import rc


def rms(a):
    return np.sqrt(np.mean(np.square(a)))


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def main_snr():
    paths = (r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB003\Data',
             r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data',
             r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB005\Data',
             r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment2\CYB101\Data',
             r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment2\CYB102\Data',)
    tasks = ['Walk','Stair']
    bg = dict()
    ecg = dict()
    for path in paths:
        print(path)
        sig_at_peak = [[] for _ in range(7)]
        sig_at_stand = [[] for _ in range(7)]
        sig_at_walk = [[] for _ in range(7)]
        for t in tasks:
            print(t)
            emg = load_emg_stack(path, task=[t])
            stack = load_dict_stack(path + '_p', task=[t])
            s_class = [np.repeat(d['step_class'], 20) for d in stack]
            for ch in range(7):
                for cl, f in zip(s_class, emg):
                    pr = peak_regs(f[7, :], 100, 200, distance=800)
                    peak_idx = np.concatenate(pr)
                    step_idx = np.concatenate((consecutive(np.where(cl==0)[0]), consecutive(np.where(cl==2)[0])))
                    step_idx = np.concatenate([st[:int(len(st)/2)] for st in step_idx])
                    buff = np.zeros_like(cl)
                    buff[step_idx] = 1
                    step_idx = buff.astype(bool)
                    stand_idx = 1 - step_idx
                    stand_idx[peak_idx] = 0
                    if np.max(stand_idx) != 0:
                        sig_at_stand[ch].append(rms(f[ch, stand_idx]))
                    else:
                        sig_at_stand[ch].append(np.nan)

                    sig_at_peak[ch].append(rms(f[ch, peak_idx]))

                    sig_at_walk[ch].append(rms(f[ch, cl != 1]))

        snr_ecg = [np.mean(np.divide(sig_at_walk[cur_ch], sig_at_peak[cur_ch])) for cur_ch in range(7)]
        snr_bg = [np.mean(np.divide(np.array(sig_at_walk[cur_ch])[np.invert(np.isnan(sig_at_stand[cur_ch]))],
                                        np.array(sig_at_stand[cur_ch])[np.invert(np.isnan(sig_at_stand[cur_ch]))]))
                  for cur_ch in range(7)]
        bg[path] = (snr_bg, np.mean(snr_bg))
        ecg[path] = (snr_ecg, np.mean(snr_ecg))
    bg['task'] = tasks
    dirp,fp, _ = incr_file(r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\EMG_filter', 'SNR', '.txt')
    print_to_file(kw_summary(
        background_snr=bg, ecg_snr=ecg).replace(',', ',\n').replace('{', '{\n').replace(': ', ':\n'), dirp+fp)
    return


def main_ecg():
    emg = load_emg_stack(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment2\CYB102\Data', task='Walk')
    sns.set_style('darkgrid')
    sns.set_context('poster')
    lat = np.empty((0, 300))
    ecg = np.empty((0, 300))
    for f in emg:
        pr = peak_regs(f[7,:], 100, 200, distance=800)
        lat = np.vstack((lat, f[4, pr]))
        ecg = np.vstack((ecg, f[7, pr]))

    to_plot = norm_emg(np.vstack([np.mean(ecg, axis=0), np.mean(lat, axis=0)]))

    fig: plt.Figure = plt.figure(constrained_layout=True)
    spec = GridSpec(ncols=2, nrows=2, figure=fig)
    ax = fig.add_subplot(spec[0,:])
    ax.plot(np.linspace(-50, 99, 300), to_plot[0, :], ls='--')
    ax.plot(np.linspace(-50, 99, 300), to_plot[1, :])
    ax.get_yaxis().set_ticklabels([" "])
    ax.set_ylabel("Arbitrary Units")
    ax.set_xlabel("Time Relative to R peak (ms)")
    plt.legend(["$\mathbb{E}\{V_{ECG}\}$", "$\mathbb{E}\{V_{Latissimus}\}$"])

    ax2 = fig.add_subplot(spec[1, 0])
    ax3 = fig.add_subplot(spec[1, 1])

    def my_spec(ts_in, ax):
        f, t, Sxx = sp.spectrogram(ts_in, 2000, nperseg=256, noverlap=128, nfft=500)
        ax.pcolormesh(t, f[:50], np.log10(np.square(Sxx[:50, :])))
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylim((0, 200))

    my_spec(emg[20][5,:], ax2)
    ax2.set_title('Latissimus Spectrogram')
    my_spec(emg[20][7,:], ax3)
    ax3.set_title('ECG Spectrogram')

    peaks, _ = sp.find_peaks(emg[20][7,:], distance=800, height=np.max(emg[20][7,:])*0.5)
    ax2.plot(peaks / 2000, np.ones_like(peaks) * 40, 'k+')
    ax3.plot(peaks / 2000, np.ones_like(peaks) * 40, 'k+')
    plt.show()
    print('yee')




def main_ar():
    sns.set_style('darkgrid')
    # emg = load_emg_stack(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data', task='Walk')
    # targ = 0
    # x0 = np.array([cur_x[targ, i*400:(i+1)*400] for cur_x in emg for i in range(int(cur_x.shape[1]/400))])
    # from statsmodels.tsa.stattools import pacf
    # pacs = [pacf(cur_seg, method='ywm') for cur_seg in x0]
    # plt.stem(np.mean(pacs, axis=0), use_line_collection=True)
    # plt.axhline(1.96 / np.sqrt(x0.shape[1]), color='red', ls='--')
    # plt.axhline(-1.96 / np.sqrt(x0.shape[1]), color='red', ls='--')
    emg = norm_emg(load_emg(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data\004_Walk12.json', task='Walk'))
    targ = 0
    ord = 4
    mu = 0.01
    x = emg[targ,:]
    X = ar_stack(x, ord)
    F = LMS(x_in=X[:,:-1], d_in=x[ord:], mu=mu)
    F.run()
    plt.plot(F.W.T)
    plt.show()


def main_anc():
    sns.set_style('darkgrid')
    sns.set_context('poster')
    emg_stack = load_emg_stack(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment2\CYB102\Data', task='Stair')
    for i in range(4,5):
        emg = emg_stack[i]
        emg = norm_emg(emg)
        s = emg[4, :]
        ref = emg[7, :]
        filt_ord = 4
        mu = 0.5
        F = anc(s=s, ref=ref, beta=1, order=filt_ord, mu=mu, eps=1/mu, pad=True, rho=0, act='tanh', pretrain=(400, 3))
        plt.figure()

        plt.plot(np.linspace(0, len(s)/2, len(s)), s*4, lw=2)
        plt.plot(np.linspace(0, len(s)/2, len(s)), ref, lw=2)
        plt.plot(np.linspace(0, len(s)/2, len(s)), F.e*4, lw=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (arbitrary units)')
        plt.legend(('Trapezius signal', 'ECG reference', 'Filtered signal'))
        plt.gca().get_yaxis().set_ticklabels([" "])
        plt.tight_layout()
        # plt.figure()
        # plt.plot(F.W.T)
    plt.show()


def main_delay_sweep():
    # region Loading and setup
    sns.set_style('darkgrid')
    n_delays = 5
    sns.set_palette(sns.cubehelix_palette(n_delays, start=.5, rot=-.75))
    pal = sns.cubehelix_palette(n_delays, start=.5, rot=-.75)
    emg = load_emg_stack(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment_Balint\CYB004\Data', task='Stair')

    # endregion
    ers_list = list()
    filters = list()
    filt_ord = 4
    targ = 5
    d_stride = 2
    n = lambda x: (x - np.mean(x)) / np.std(x)
    mu = 0.005
    rec = 0
    for rec in range(10):
        print(rec)
        segs = np.concatenate(peak_regs(n(emg[rec][7, :]), 100, 200, distance=800))
        # filters = [anc(n(emg[rec][targ, :]), n(emg[rec][7, :]), 0.001, rho=1e-7, alph=0.99, alg='A&F', mode='GASS', delay=d, order=filt_ord) for d in range(0,n_delays*d_stride,d_stride)]
        # filters = [anc(n(emg[rec][targ, :]), n(emg[rec][7, :]), mu, rho=0.15, eps=1/mu, mode='GNGD', delay=d, order=filt_ord, act='tanh', scale=4, bias=0.1, reps=1, sign=True) for d in range(0,n_delays*d_stride,d_stride)]
        filters = [anc(n(emg[rec][targ, :]), n(emg[rec][7, :]), 0.004/3,  mode='LMS', delay=d, order=filt_ord, act='tanh', scale=2.5, bias=0.01, reps=1) for d in range(0, n_delays * d_stride, d_stride)]
        # filters = [anc(n(emg[rec][targ, :]), n(emg[rec][7, :]), 0.001, mode='LMS', delay=d, order=filt_ord, reps=1) for d in range(0, n_delays * d_stride, d_stride)]
        delay = 0
        ers = list()
        for f in filters:
            ers.append(np.mean(np.abs(f.e[segs - (filt_ord + delay - 1)] ** 2)))
            delay += d_stride
        ers_list.append(ers)
    delay = 0
    for f in filters:
        plt.figure(1)
        plt.plot(np.arange(delay,delay+len(np.squeeze(f.e))), np.squeeze(f.e).T, zorder=delay*2)
        plt.figure(2)
        plt.plot(np.arange(delay, delay + f.W.shape[1]), f.W.T, zorder=delay * 2, color=pal[int(delay/d_stride)])
        delay += d_stride

    plt.figure(1)
    plt.plot(np.arange(-filt_ord+1, -filt_ord+1 + len(n(emg[rec][7, :]))), n(emg[rec][7, :]) / 3, color='red', alpha=0.8, zorder=-5)
    plt.plot(np.arange(-filt_ord+1, -filt_ord+1 + len(n(emg[rec][targ, :]))), n(emg[rec][targ, :]), color='orange', alpha=0.8, zorder=-5)
    plt.figure()
    plt.plot(np.arange(0,n_delays*2,2), np.mean(ers_list, 0), c=pal[int(len(pal)/2)])
    plt.ylabel('MSE'),
    plt.xlabel('delay (samples)')
    plt.title('Trap, GNGD, sign')


    plt.show()
    return


def peak_regs(ecg, before, after, **kwargs):
    height=np.max(ecg)*0.6
    peaks, _ = sp.find_peaks(ecg, height=height, **kwargs)
    if min(peaks)<before:
        peaks = peaks[1:]
    if max(peaks)>len(ecg)-after:
        peaks = peaks[:-1]
    return [np.arange(peak-before, peak+after) for peak in peaks]


def crop_to_peaks(emg, ecg, order, delay):
    X = ar_stack(ecg, order)
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
    emg = load_emg(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB005\Data\005_Walk20.json')
    emg = norm_emg(emg)
    step_dict = load_dict(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB005\Data_p\005_ParamWalk20.json')

    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    cl = np.array(step_dict["step_class"])
    step_idx = consecutive(np.where(cl == 0)[0])

    peaks, _ = sp.find_peaks(emg[7, :], distance=800, height=2.2)
    targ = 6



    plt.plot(emg[targ, :])
    plt.figure()
    n_bins = 1000
    F = spectrum_lms(emg[targ, :], n_bins, gamma=0.01)
    #spec = norm_emg(np.abs(F.W[1:int(F.W.shape[0]/4), 3:]).T).T
    #spec = spec-np.min(spec)+1
    import seaborn as sns
    sns.set_style('darkgrid')
    sns.set_context('poster')
    spec = np.log10(np.square(np.abs(F.W[1:int(F.W.shape[0] / 8), 1::2])))
    plt.figure(figsize=(8,5))
    plt.pcolormesh(np.linspace(0, spec.shape[1]/1000, spec.shape[1]), np.arange(spec.shape[0])/n_bins*2000, spec)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.vlines([st[0]/100 for st in step_idx], ymin=0, ymax=(spec.shape[0]-1)/n_bins*2000)
    plt.vlines([p / 2000for p in peaks], ymin=0, ymax=(spec.shape[0] - 1) / n_bins * 2000, lw=0.5, color='w' )
    plt.show()
    return


def main_spectrum_visu():
    # region Loading and setup
    sns.set_style('darkgrid')
    sns.set_context('poster')
    emg = load_emg(r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB005\Data\005_Walk05.json', task='')
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
        axs[0].set_title('Trapezius Spectrum')
        my_spec(emg1, axs[1])
        axs[1].set_title('Erector Spinae Spectrum')
        my_spec(ecg, axs[2])
        axs[2].set_title('ECG Spectrum')

        # endregion

        # region ECG ARMA modelling

        peaks, _ = sp.find_peaks(ecg, distance=800, height=2.2)
        if min(peaks) < 59:
            peaks = peaks[1:]
        if max(peaks) > len(ecg) - 202:
            peaks = peaks[:-1]
        axs[2].plot(peaks / 2000, np.ones_like(peaks) * 100, 'k+')
        axs[0].plot(peaks / 2000, np.ones_like(peaks) * 100, 'k+')
        axs[1].plot(peaks / 2000, np.ones_like(peaks) * 100, 'k+')
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
        plt.stem( np.mean(pacs, axis=0), use_line_collection=True)
        a = plt.axhline(1.96 / np.sqrt(qrs.shape[1]), color='tab:red', ls='--', label='5% significance')
        plt.axhline(-1.96 / np.sqrt(qrs.shape[1]), color='tab:red', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.xlabel('Delay (sample)')
        plt.ylabel('Partial Autocorrelation')
        plt.title('ECG Partial Autocorrelation Function')
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
    main_anc()
