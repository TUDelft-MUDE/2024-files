import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

_, H, T = np.genfromtxt('dataset_HT.csv', delimiter=",", unpack=True, skip_header=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(H,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Wave height, H (m)')
ax[0].grid()

ax[1].plot(T,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Water period, T (s)')
ax[1].grid()

print(stats.describe(H))

print(stats.describe(T))

def ecdf(YOUR_INPUTS):
    #your code
    return YOUR_OUTPUT

rs_H = #your code here
rs_T = #your code here

rs_Fh = #your code here

Fh = #your code here

fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_H, rs_T, 40, 'k', label = 'Simulations')
axes.scatter(H, T, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Wave height, H (m)')
axes.set_ylabel('Wave period, T (s)')
axes.legend()
axes.grid()

