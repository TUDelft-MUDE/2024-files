# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper import plot_contour
from helper import Bivariate

# ----------------------------------------
# YOUR_CODE_HERE # many lines
# parameters1 = st.norm.fit_loc_scale(data_x1)
# dist_x1 = st.norm(*parameters1)

# ----------------------------------------
# YOUR_CODE_HERE # many lines

# ----------------------------------------
# plot_contour? # uncomment and run to read docstring

# ----------------------------------------
bivar_dist = Bivariate(YOUR_CODE_HERE, YOUR_CODE_HERE, YOUR_CODE_HERE)

plot_contour(YOUR_CODE_HERE, YOUR_CODE_HERE, data=data)

# ----------------------------------------
region_example = np.array([[0, 5, 12, 20, 28, 30],
                           [5, 20, 0, 18, 19, 12]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[23, 13],
             region=region_example);


# ----------------------------------------
# YOUR_CODE_HERE
# DEFINITELY more than one line.
# probably several cells too ;)

# ----------------------------------------
# YOUR_CODE_HERE
# DEFINITELY more than one line.
# probably several cells too ;)

# ----------------------------------------
plot_values = np.linspace(sample_Z.min(), sample_Z.max(), 30)
fig, ax = plt.subplots(1)
ax.hist([YOUR_CODE_HERE, YOUR_CODE_HERE],
         bins=plot_values,
         density=True,
         edgecolor='black');
ax.legend(['Data', 'MCS Sample'])
ax.set_xlabel('$Z(X_1,X_2)$')
ax.set_xlabel('Empirical Density [--]')
ax.set_title('Comparison of Distributions of $Z$');

# ----------------------------------------
def ecdf(var):
    x = np.sort(var) # sort the values from small to large
    n = x.size # determine the number of datapoints
    y = np.arange(1, n+1) / (n + 1)
    return [y, x]


# ----------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(8, 5))

axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE, 
          color='k', label='Data')
axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE,
          color='r', label='MCS Sample')
axes.set_xlabel('$Z(X_1,X_2)$')
axes.set_ylabel('CDF, $\mathrm{P}[Z < z]$')
axes.set_title('Comparison: CDF (log scale expands lower tail)')
axes.set_yscale('log')
axes.legend()
axes.grid()

# ----------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(8, 5))

axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE, 
          color='k', label='Data')
axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE,
          color='r', label='MCS Sample')
axes.set_xlabel('$Z(X_1,X_2)$')
axes.set_ylabel('Exceedance Probability, $\mathrm{P}[Z > z]$')
axes.set_title('Comparison: CDF (log scale expands upper tail)')
axes.set_yscale('log')
axes.legend()
axes.grid()

