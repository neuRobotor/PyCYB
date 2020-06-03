import scipy as sp
import numpy as np
from scipy import signal


def norm_emg(data):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def denoise(emg, sfreq=2000, high_band=20, low_band=450):
    emg = emg - np.mean(emg)
    emg = emg - np.mean(emg)
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    ba = sp.signal.butter(4, [high_band, low_band], btype='bandpass', output='ba')
    b, a = sp.signal.iirnotch(50, fs=sfreq, Q=30)
    emg_notched = sp.signal.filtfilt(b, a, emg)
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(ba[0], ba[1], emg_notched)
    return emg_filtered

def envelope(emg, low_pass=10, sfreq=2000, high_band=20, low_band=450):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    emg = emg - np.mean(emg)
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    ba = sp.signal.butter(4, [high_band, low_band], btype='bandpass', output='ba')
    b,a = sp.signal.iirnotch(50, fs=sfreq, Q=30)
    emg_notched = sp.signal.filtfilt(b, a, emg)
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(ba[0], ba[1], emg_notched)

    # process EMG signal: rectify
    emg_rectified = np.abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass / (sfreq/2)
    ba = sp.signal.butter(4, low_pass, btype='lowpass', output='ba')
    emg_envelope = sp.signal.filtfilt(ba[0], ba[1], emg_rectified)

    return emg_envelope


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y