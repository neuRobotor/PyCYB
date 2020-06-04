import numpy as np
from scipy import signal
from EMG_filter.lms_filt import anc
from EMG_filter.lms_filt import spectrum_lms


def smooth(data):
    return np.apply_along_axis(lambda d: signal.medfilt(d, kernel_size=21), axis=1, arr=data)


def lms_anc(data, clear_ecg=False, filter_mask=None, **kwargs):
    data = norm_emg(data)
    filter_mask = np.arange(len(data)) if filter_mask is None else filter_mask
    for ch in filter_mask:
        s = data[ch]
        ref = data[-1]
        F = anc(s=s, ref=ref, **kwargs)
        data[ch] = (F.e)
    if clear_ecg:
        data = data[1:-1]
    return data

def norm_emg(data, **kwargs):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def bp_filter(data, high_band=7, low_band=400, sfreq=2000, filt_ord=2, causal=True, **kwargs):
    data = norm_emg(data)
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)
    # create bandpass filter for EMG
    b, a = signal.butter(filt_ord, [high_band, low_band], btype='bandpass', output='ba')
    # process EMG signal: filter EMG
    return signal.lfilter(b,a, data, axis=1) if causal else signal.filtfilt(b,a, data, axis=1)


def spec_proc(data, fbins=500, gamma=0.01, frange=(0, 500), fsamp=2000):
    data = norm_emg(data)
    return np.array([np.abs(
        spectrum_lms(emg, fbins, gamma=gamma).W[int(fbins*frange[0]/fsamp):int(fbins*frange[1]/fsamp), 1:]).T
                     for emg in data])

