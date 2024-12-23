# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

# ----------------------------------------
# Import
data = np.genfromtxt('dataset_concrete.csv', delimiter=",", skip_header=True)

# Clean
data = data[~np.isnan(data)]

# plot time series
plt.figure(figsize=(10, 6))
plt.plot(data,'ok')
plt.xlabel('# observation')
plt.ylabel('Concrete compressive strength [MPa]')
plt.grid()

weights = 5*np.ones(len(data))
plt.hist(data, orientation='horizontal', weights=weights, color='lightblue', rwidth=0.9)

# ----------------------------------------
# Statistics

df_describe = pd.DataFrame(data)
df_describe.describe()

# ----------------------------------------
# def ecdf(YOUR_CODE_HERE):
#     YOUR_CODE_HERE # may be more than one line
#     return YOUR_CODE_HERE

# SOLUTION:
def ecdf(observations):
    x = np.sort(observations)
    n = x.size
    y = np.arange(1, n+1) / (n + 1)
    return [y, x]

# ----------------------------------------
# loc = YOUR_CODE_HERE
# scale = YOUR_CODE_HERE

# fig, axes = plt.subplots(1, 1, figsize=(10, 5))
# axes.hist(YOUR_CODE_HERE,
#           edgecolor='k', linewidth=0.2, color='cornflowerblue',
#           label='Empirical PDF', density = True)
# axes.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
#           'k', linewidth=2, label='YOUR_DISTRIBUTION_NAME_HERE PDF')
# axes.set_xlabel('Compressive strength [MPa]')
# axes.set_title('PDF', fontsize=18)
# axes.legend()

# SOLUTION
loc = 28.167
scale = 13.097

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.hist(data,
          edgecolor='k', linewidth=0.2, color='cornflowerblue',
          label='Empirical PDF', density = True)
axes.plot(np.sort(data), stats.gumbel_r.pdf(np.sort(data), loc, scale),
          'k', linewidth=2, label='Gumbel PDF')
axes.set_xlabel('Compressive strength [MPa]')
axes.set_title('PDF', fontsize=18)
axes.legend()
fig.savefig('pdf.svg')

# ----------------------------------------
# fig, axes = plt.subplots(1, 1, figsize=(10, 5))

# axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE, 
#           color='k', label='Empirical CDF')
# axes.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
#           color='cornflowerblue', label='YOUR_DISTRIBUTION_NAME_HERE CDF')
# axes.set_xlabel('Compressive strength [MPa]')
# axes.set_ylabel('${P[X > x]}$')
# axes.set_title('Exceedance plot in log-scale', fontsize=18)
# axes.set_yscale('log')
# axes.legend()
# axes.grid()

# SOLUTION
fig, axes = plt.subplots(1, 1, figsize=(10, 5))

axes.step(ecdf(data)[1], 1-ecdf(data)[0], 
          color='k', label='Empirical CDF')
axes.plot(ecdf(data)[1], 1-stats.gumbel_r.cdf(ecdf(data)[1], loc, scale),
          color='cornflowerblue', label='Gumbel CDF')
axes.set_xlabel('Compressive strength [MPa]')
axes.set_ylabel('${P[X > x]}$')
axes.set_title('Exceedance plot in log-scale', fontsize=18)
axes.set_yscale('log')
axes.legend()
axes.grid()
fig.savefig('cdf.svg')

# ----------------------------------------
# fig, axes = plt.subplots(1, 1, figsize=(10, 5))

# axes.plot([0, 120], [0, 120], 'k')
# axes.scatter(YOUR_CODE_HERE, YOUR_CODE_HERE, 
#              color='cornflowerblue', label='Gumbel')
# axes.set_xlabel('Observed compressive strength [MPa]')
# axes.set_ylabel('Estimated compressive strength [MPa]')
# axes.set_title('QQplot', fontsize=18)
# axes.set_xlim([0, 120])
# axes.set_ylim([0, 120])
# axes.set_xticks(np.arange(0, 121, 20))
# axes.grid()

# SOLUTION
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

axes.plot([0, 120], [0, 120], 'k')
axes.scatter(ecdf(data)[1], stats.gumbel_r.ppf(ecdf(data)[0], loc, scale), 
             color='cornflowerblue', label='Gumbel')
axes.set_xlabel('Observed compressive strength [MPa]')
axes.set_ylabel('Estimated compressive strength [MPa]')
axes.set_title('QQplot', fontsize=18)
axes.set_xlim([0, 120])
axes.set_ylim([0, 120])
axes.set_xticks(np.arange(0, 121, 20))
axes.grid()
fig.savefig('ppf.svg')

