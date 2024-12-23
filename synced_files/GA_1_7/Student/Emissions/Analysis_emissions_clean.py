# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

# ----------------------------------------
# Import
C, H = np.genfromtxt('dataset_traffic.csv', delimiter=",", unpack=True, skip_header=True)

# plot time series
fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(H,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Number of heavy vehicles, H')
ax[0].grid()

ax[1].plot(C,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Number of cars, C')
ax[1].grid()

# ----------------------------------------
# Statistics for H

print(stats.describe(H))

# ----------------------------------------
# Statistics for d

print(stats.describe(C))

# ----------------------------------------
def ecdf(YOUR_INPUT):
    #Your code
    return YOUR_OUTPUT

# ----------------------------------------
#Your plot here

# ----------------------------------------
#your code here

# ----------------------------------------
#Your code here

# ----------------------------------------
# Here, the solution is shown for the Lognormal distribution

# Draw random samples
rs_H = #your code here
rs_C = #your code here

#Compute Fh
rs_CO2 = #your code here

#repeat for observations
CO2 = #your code here

#plot the PDF and the CDF

# ----------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_H, rs_C, 40, 'k', label = 'Simulations')
axes.scatter(H, C, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Number of heavy vehicles, H ')
axes.set_ylabel('Number of cars, C')
axes.legend()
axes.grid()

# ----------------------------------------
#Correlation coefficient calculation here

