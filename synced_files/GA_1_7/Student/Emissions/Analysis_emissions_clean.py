import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

C, H = np.genfromtxt('dataset_traffic.csv', delimiter=",", unpack=True, skip_header=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(H,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Number of heavy vehicles, H')
ax[0].grid()

ax[1].plot(C,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Number of cars, C')
ax[1].grid()

print(stats.describe(H))

print(stats.describe(C))

def ecdf(YOUR_INPUT):
    #Your code
    return YOUR_OUTPUT

rs_H = #your code here
rs_C = #your code here

rs_CO2 = #your code here

CO2 = #your code here

fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_H, rs_C, 40, 'k', label = 'Simulations')
axes.scatter(H, C, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Number of heavy vehicles, H ')
axes.set_ylabel('Number of cars, C')
axes.legend()
axes.grid()

