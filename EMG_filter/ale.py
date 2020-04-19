from numpy import sqrt, arange, sign, cos, pi, zeros_like, zeros, hstack, array, log10
from numpy.random import randn
from scipy import signal
from matplotlib.pyplot import plot, xlabel, ylabel, grid, title, xlim, ylim, legend, figure, show
from utility.save_load_util import load_emg
import numpy as np


def norm_emg(data):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]

def lms_ale(M, mu, x=None, sqwav=False, SNR=10, N=1000):
    """
    lms_ale lms ALE adaptation algorithm using an IIR filter.
    n,x,x_hat,e,ao,F,Ao = lms_ale(SNR,N,M,mu)

    *******LMS ALE Simulation************
    SNR = Sinusoid SNR in dB
    N = Number of simulation samples
    M = FIR Filter length (order M-1)
    mu = LMS step-size
    mode = 0 <=> sinusoid, 1 <=> squarewave

    n = Index vector
    x = Noisy input
    x_hat = Filtered output
    e = Error sequence
    ao = Final value of weight vector
    F = Frequency response axis vector
    Ao = Frequency response of filter in dB
    **************************************
    Mark Wickert, November 2014
    """
    if x is None:
        # Sinusoid SNR = (A^2/2)/noise_var
        n = arange(0,N+1) # length N+1
        if not sqwav:
            x = 1*cos(2*pi*1/20*n) # Here A = 1, Fo/Fs = 1/20
            x += sqrt(1/2/(10**(SNR/10)))*randn(N+1)
        else: # Squarewave case
            x = 1*sign(cos(2*pi*1/20*n)) # Here A = 1, Fo/Fs = 1/20
            x += sqrt(1/1/(10**(SNR/10)))*randn(N+1)
    else:
        n = arange(0, len(x))  # length N+1
    # Normalize mu
    mu /= M + 1

    # White Noise -> Delta = 1, so delay x by one sample
    delay = [0,]*10
    delay.extend([1])
    y = signal.lfilter(delay, 1, x)
    # Initialize output vector x_hat to zero
    x_hat = zeros_like(x)
    # Initialize error vector e to zero
    e = zeros_like(x)
    # Initialize weight vector to zero
    ao = zeros(M + 1)
    # Initialize filter memory to zero
    zi = signal.lfiltic(ao, 1, y=0)
    # Initialize a vector for holding ym of length M+1
    ym = zeros_like(ao)
    for k, yk in enumerate(y):
        # Filter one sample at a time
        x_hat[k], zi = signal.lfilter(ao, 1, [yk], zi=zi)
        # Form the error sequence
        e[k] = x[k] - x_hat[k]
        # Update the weight vector
        ao = ao + 2 * mu * e[k] * ym
        # Update vector used for correlation with e[k]
        ym = hstack((array(yk), ym[0:-1]))
        # Create filter frequency response
    F = arange(0, 0.5, 1 / 512)
    w, Ao = signal.freqz(ao, 1, 2 * pi * F)
    Ao = 20 * log10(abs(Ao))
    return n, x, x_hat, e, ao, F, Ao


def main():
    figure()
    x = load_emg(r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data\004_Stair03.json', task='Stair')
    x = norm_emg(x)
    x = x[3,:]
    n,x,x_hat,e,ao,F,Ao = lms_ale(30,0.005,x=x, sqwav=False)
    plot(n,e**2)
    ylabel(r'MSE')
    xlabel(r'Time Index n')
    title('SNR = 10 dB: Sinewave')
    grid()

    figure()
    plot(n,x, lw=0.5)
    plot(n,x_hat, lw=0.5)
    plot(n,e, lw=0.5)
    legend((r'$x[n]$',r'$\hat{x}[n]$',r'$e[n]$'),loc='best')
    ylabel(r'Input/Output Signal')
    xlabel(r'Time Index n')
    title('SNR = 10 dB: Sinewave')
    grid()

    figure()
    plot(F,Ao)
    plot([0.05,0.05],[-40,0],'r')
    ylabel(r'Frequency Response in dB')
    xlabel(r'Normalized Frequency $\omega/(2\pi)$')
    title('SNR = 10 dB: Sinewave')
    grid()

    show()


if __name__ == "__main__":
    main()
