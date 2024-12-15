
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

h, u = np.genfromtxt('dataset_hu.csv', delimiter=",", unpack=True, skip_header=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(h,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Water depth, h (m)')
ax[0].grid()

ax[1].plot(u,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Water velocity, u (m/s)')
ax[1].grid()

print(stats.describe(h))

print(stats.describe(u))

def ecdf(YOUR_INPUT:
    #Your code
    return YOUR_OUTPUT

rs_h = #Your code here
rs_u = #Your code here

rs_q = #Your code here

q = #Your code here

fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_h, rs_u, 40, 'k', label = 'Simulations')
axes.scatter(h, u, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Wave height, H (m)')
axes.set_ylabel('Wave period, T (s)')
axes.legend()
axes.grid()

