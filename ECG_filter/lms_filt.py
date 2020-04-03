from ECG_filter.lms import *
import numpy as np
import scipy.signal as sp


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    b = [1, 0.4, 0, 0.2]
    n = np.random.randn(10000)
    eta = sp.lfilter(b, [1], n)
    d = np.sin(np.linspace(0, 4 * np.pi, 10000))
    s = d + eta
    n = n[1000:]
    s = s[1000:]
    y, e, w = anc(s, n, 0.005, order=4, eps=0.1)
    plt.plot(s)
    plt.plot(e.T)
    plt.figure()
    plt.plot(np.squeeze(w).T)
    plt.show()
    return


def fourier_mat(n_row, n_col):
    return np.array([np.exp(1j*2*np.pi/n_row*np.arange(n_col)*row) for row in np.arange(n_row)])


def spectrum_lms(d, n_bins, **kwargs):
    X = fourier_mat(n_bins, len(d))
    F = CLMS(X, d, 1, **kwargs)
    F.run()
    return F



def anc(s, ref, mu, order=1, delay=0, mode='GNGD', filter_in=None, **kwargs):
    X = lms_stack(ref, order)
    X = X[:, :X.shape[1] - delay]
    s = s[order + delay - 1:]
    if filter_in is not None:
        F = filter_in
    elif mode is 'GNGD':
        F = GNGD(X, s, mu, **kwargs)
    elif mode is 'GASS':
        F = GASS(X, s, mu, **kwargs)
    else:
        F = LMS(X, s, mu, **kwargs)
    F.run()
    return F


if __name__ == "__main__":
    main()
