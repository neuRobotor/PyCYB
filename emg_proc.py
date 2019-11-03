import scipy as sp
import numpy as np
from scipy import signal


def envelope(emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
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

    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(ba[0], ba[1], emg)

    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass / sfreq
    ba = sp.signal.butter(4, low_pass, btype='lowpass', output='ba')
    emg_envelope = sp.signal.filtfilt(ba[0], ba[1], emg_rectified)

    return emg_envelope
