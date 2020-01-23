import numpy as np
from numpy import pi
from scipy import signal
import matplotlib.pyplot as plt
from utility.load_util import load_emg
from utility.emg_proc import *
import seaborn as sns
from ale import lms_ale


def norm_emg(data):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def lms_ic(M, s, y=None, delta=1, mu=0.005):
    """
    Adaptive Interference Cancellation
    ECE 5655/4655 Real-Time DSP 8–15
    Adaptive interference canceller using LMS and FIR
    n,x,x_hat,e,ao,F,Ao = lms_ic(s,SIR,N,M,delta,mu)

    *******LMS Interference Cancellation Simulation************
    s = Input speech signal
    SIR = Speech signal power to sinusoid interference level in dB
    N = Number of simulation samples
    M = FIR Filter length (order M-1)
    delta = Delay used to generate the reference signal
    mu = LMS step-size

    n = Index vector
    x = Noisy input
    x_hat = Filtered output
    e = Error sequence
    ao = Final value of weight vector
    F = Frequency response axis vector
    Ao = Frequency response of filter
    **************************************
    Mark Wickert, November 2014
    """
    n = np.arange(0, len(s))
    if y is None:
        # Form the reference signal y via delay delta
        y = signal.lfilter(np.hstack((np.zeros(delta), [1])), 1, s)
    else:
        #y = signal.lfilter(np.hstack((np.zeros(delta), [1])), 1, s)
        pass
    mu /= M + 1
    # Initialize output vector x_hat, e, and filter weights and memory, and correlation memory to zero
    x_hat = np.zeros_like(s)
    e = np.zeros_like(s)
    ao = np.zeros(M+1)
    zi = signal.lfiltic(ao,1,y=0)
    ym = np.zeros_like(ao)
    for k,yk in enumerate(y):
        # Filter one sample at a time
        x_hat[k],zi = signal.lfilter(ao,1,np.array([yk]),zi=zi)
        # Form the error sequence
        e[k] = s[k] - x_hat[k]
        # Update the weight vector
        ao = ao + 2*mu*e[k]*ym
        # Update vector used for correlation with e[k]
        ym = np.hstack((np.array(yk), ym[0:-1]))
    # Create filter frequency response
    F = np.arange(0,0.5,1/512)
    w,Ao = signal.freqz(ao,1,2*pi*F)
    Ao = 20*np.log10(abs(Ao))
    return n,s,x_hat,e,ao,F,Ao

sns.set_style('darkgrid')
s = load_emg(r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data', task='Stair')
s = norm_emg(s)
s = s[:,:int(s.shape[1]/20)]
y = s[7, :]
s = s[5, :]

n,x,x_hat,e,ao,F,Ao = lms_ic(8, s, y, mu=0.005)

print(np.mean(e**2))
plt.plot(n,e**2)
plt.ylabel(r'MSE')
plt.xlabel(r'Time Index n')

plt.figure()
plt.plot(n,x)
plt.plot(n,y)
plt.plot(n,e)
plt.plot(n,x_hat)
plt.legend((r'$x[n]$',r'$\hat{x}[n]$',r'$e[n]$'),loc='best')
plt.ylabel(r'Input/Output Signal')
plt.xlabel(r'Time Index n')

plt.figure()
plt.plot(F,Ao)
plt.ylabel(r'Frequency Response in dB')
plt.xlabel(r'Normalized Frequency $\omega/(2\pi)$')

plt.show()

n,x,x_hat,e,ao,F,Ao = lms_ale(30,0.005,x=e)

plt.figure()
plt.plot(n,x)
plt.plot(n,x_hat)
plt.plot(n,e)
plt.legend((r'$x[n]$',r'$\hat{x}[n]$',r'$e[n]$'),loc='best')
plt.ylabel(r'Input/Output Signal')
plt.xlabel(r'Time Index n')
n,x,x_hat,e,ao,F,Ao = lms_ale(30,0.005,x=x_hat)
plt.figure()
plt.plot(n,x)
plt.plot(n,x_hat)
plt.plot(n,e)
plt.legend((r'$x[n]$',r'$\hat{x}[n]$',r'$e[n]$'),loc='best')
plt.ylabel(r'Input/Output Signal')
plt.xlabel(r'Time Index n')

plt.show()