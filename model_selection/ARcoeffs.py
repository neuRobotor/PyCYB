from spectrum import aryule
from utility.save_load_util import load_emg
from utility.emg_proc import norm_emg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ECG_filter.anc import lms_ic


def main():
    sns.set_style('darkgrid')
    s = load_emg(r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data', task='Walk')
    s = norm_emg(s)
    # def f(x_in):
    #     n, x, x_hat, e, ao, F, Ao = lms_ic(3, x_in, s[7,:], mu=0.01)
    #     return e
    # s_filt = np.apply_along_axis(f, 1, s[0:7, :])
    # s[0:7, :] = s_filt

    PACs = list()
    count = 0
    for sig in s:
        count += 1
        print(count)
        PAC = list()
        _, _, coeff_reflection = aryule(sig, 15)
        PACs.append(coeff_reflection)

    fig, axes = plt.subplots(4, 2)
    fig.suptitle("Partial Autocorrelation Functions")
    axes = axes.flatten()
    muscle_names = ['L Internal Oblique', 'R Internal Oblique', 'L External Oblique', 'R External Oblique',
                    'Latissimus Dorsi', 'Transverse Trapezius', 'Erector Spinae', 'ECG']
    for i, ax in enumerate(axes):
        x, _, _ = ax.stem(np.arange(6, 16), PACs[i][5:], use_line_collection=True, basefmt='grey')
        x.set_label(muscle_names[i])
        l = ax.axhline(1.96 / np.sqrt(s.shape[1]), color='tab:orange', ls='--')
        l.set_label('95% confidence interval')
        ax.axhline(-1.96 / np.sqrt(s.shape[1]), color='tab:orange', ls='--')
        if i%2 is 0:
            ax.set_ylabel('PAC')
        if i<6:
            ax.set_xticks(np.arange(6,16,2))
            ax.set_xticklabels([])
        else:
            ax.set_xticks(np.arange(6, 16, 2))
            ax.set_xlabel("Lag")
        x.set_markersize(4)
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()



if __name__ == '__main__':
    main()