import numpy as np
from scipy.stats import norm, chi
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


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


data_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB005\Data'
with open(data_path+r'\params.csv') as f:
    reader = csv.reader(f)
    params = list(reader)
    params = [[float(el) for el in row] for row in params]
for i, p in enumerate(params):
    if i != 6 and i != 0:
        continue
    pdf = make_pdf(chi, p)
    pdf.plot(lw=2)

print(chi.cdf(8.47, loc=params[6][-2], scale=params[6][-1], *params[6][:-2]))
# Display
plt.show()