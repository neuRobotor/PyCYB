import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats._continuous_distns import _distn_names
import seaborn as sns
import os
import json
DISTRIBUTIONS = _distn_names
DISTRIBUTIONS = ['f', 'chi', 'chi2', 'foldnorm', 'halfnorm', 'invgauss']
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
sns.set()

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data

    #data = np.abs(data)
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0


    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for dist in DISTRIBUTIONS:
        print(dist)
        distribution = getattr(st, dist)
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y[y!=0] - pdf[y!=0], 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax, label=dist)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


# Load data from statsmodels datasets
data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB005\Data'
n_channels = 8
X = np.empty((n_channels, 0))
for i, file in enumerate(sorted([f for f in os.listdir(data_path) if f.endswith('.json')])):
    with open(data_path+'\\'+file) as json_file:
        dict_data = json.load(json_file)
        X = np.concatenate((X, dict_data["EMG"]), axis=1)
    if i>=90:
        break
data = pd.Series(X[0, :])
data = (data - np.mean(data))/np.std(data)
#data = np.abs(data[abs(data - np.mean(data)) < 9 * np.std(data)])
data = np.abs(data)
data.plot(kind='hist', bins=50, density=True, alpha=0.5)
# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=200, density=True, alpha=0.5)
# Save plot limits
dataYLim = ax.get_ylim()
plt.ticklabel_format(style='sci', scilimits=(0,0))
# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 500, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'sEMG histogram\n All Fitted Distributions')
ax.set_xlabel(u'Voltage(V)')
ax.set_ylabel('Frequency')
plt.legend()

# Make PDF with best params
print("Making PDF")
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
plt.legend()
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.5f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)
plt.ticklabel_format(style='sci', scilimits=(0,0))
ax.set_title(u'sEMG with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Voltage (V)')
ax.set_ylabel('Frequency')
plt.show()