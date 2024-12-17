
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

def ecdf(var):
    x = np.sort(var) # sort the values from small to large
    n = x.size # determine the number of datapoints
    y = np.arange(1, n+1) / (n+1)
    return [y, x]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(H, edgecolor='k', linewidth=0.2, 
             color='cornflowerblue', label='Wave height, H (m)', density = True)
axes[0].set_xlabel('Random variable (X)')
axes[0].set_ylabel('density')
axes[0].hist(T, edgecolor='k', linewidth=0.2, alpha = 0.5, 
             color='grey', label='Wave period, T (s)', density = True)
axes[0].set_title('PDF', fontsize=18)
axes[0].grid()
axes[0].legend()

axes[1].step(ecdf(H)[1], ecdf(H)[0], 
             color='cornflowerblue', label='Wave height, H (m)')
axes[1].set_xlabel('Random variable (X)')
axes[1].set_ylabel('${P[X \leq x]}$')
axes[1].step(ecdf(T)[1], ecdf(T)[0], 
             color='grey', label='Wave period, T (s)')
axes[1].set_title('CDF', fontsize=18)
axes[1].legend()
axes[1].grid()

params_H = stats.expon.fit(H, floc=0)
params_T = stats.gumbel_r.fit(T)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].step(ecdf(H)[1], 1-ecdf(H)[0], 
             color='k', label='Wave height, H')
axes[0].plot(ecdf(H)[1], 1-stats.expon.cdf(ecdf(H)[1], *params_H),
             '--', color = 'grey', label='Exponential')
axes[0].set_xlabel('Wave height, H (m)')
axes[0].set_ylabel('${P[X > x]}$')
axes[0].set_title('H', fontsize=18)
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid()

axes[1].step(ecdf(T)[1], 1-ecdf(T)[0], 
             color='k', label='Wave period, T')
axes[1].plot(ecdf(T)[1], 1-stats.gumbel_r.cdf(ecdf(T)[1], *params_T),
             '--', color = 'grey', label='Gumbel')
axes[1].set_xlabel('Wave period, T (s)')
axes[1].set_ylabel('${P[X > x]}$')
axes[1].set_title('T', fontsize=18)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot([trunc(min(H)), ceil(max(H))], [trunc(min(H)), ceil(max(H))], 'k')
axes[0].scatter(ecdf(H)[1], stats.expon.ppf(ecdf(H)[0], *params_H), 
             color='grey', label='Exponential')
axes[0].set_xlabel('Observed H (m)')
axes[0].set_ylabel('Estimated H (m)')
axes[0].set_title('H', fontsize=18)
axes[0].set_xlim([trunc(min(H)), ceil(max(H))])
axes[0].set_ylim([trunc(min(H)), ceil(max(H))])
axes[0].legend()
axes[0].grid()

axes[1].plot([trunc(min(T)), ceil(max(T))], [trunc(min(T)), ceil(max(T))], 'k')
axes[1].scatter(ecdf(T)[1], stats.gumbel_r.ppf(ecdf(T)[0], *params_T), 
             color='grey', label='Gumbel')
axes[1].set_xlabel('Observed T (s)')
axes[1].set_ylabel('Estimated T (s)')
axes[1].set_title('T', fontsize=18)
axes[1].set_xlim([trunc(min(T)), ceil(max(T))])
axes[1].set_ylim([trunc(min(T)), ceil(max(T))])
axes[1].legend()
axes[1].grid()

_, p_H = stats.kstest(H,stats.expon.cdf, args=params_H)
_, p_T = stats.kstest(T,stats.gumbel_r.cdf, args=params_T)

print('The p-value for the fitted Gumbel distribution to H is:', round(p_H, 3))
print('The p-value for the fitted Uniform distribution to d is:', round(p_T, 3))

rs_H = stats.expon.rvs(*params_H, size = 10000)
rs_T = stats.gumbel_r.rvs(*params_T, size = 10000)

rs_Fh = 255.4 * rs_H * rs_T**2 - 490.4*rs_T**2

Fh = 255.4 * H * T**2 - 490.4*T**2

fig, axes = plt.subplots(1, 2, figsize=(12, 7))
axes[0].hist(rs_Fh, edgecolor='k', linewidth=0.2, density = True, label = 'From simulations')
axes[0].hist(Fh, edgecolor='k', facecolor = 'orange', alpha = 0.5, linewidth=0.2, 
             density = True, label = 'From observations')
axes[0].set_xlabel('Horizontal force (kN)')
axes[0].set_ylabel('density')
axes[0].set_title('PDF', fontsize=18)
axes[0].legend()
axes[0].grid()

axes[1].step(ecdf(rs_Fh)[1], 1-ecdf(rs_Fh)[0], label = 'From simulations')
axes[1].step(ecdf(Fh)[1], 1-ecdf(Fh)[0], color = 'orange', label = 'From observations')
axes[1].set_xlabel('Horizontal force (kN)')
axes[1].set_ylabel('${P[X > x]}$')
axes[1].set_title('Exceedance plot', fontsize=18)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid()

fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_H, rs_T, 40, 'k', label = 'Simulations')
axes.scatter(H, T, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Wave height, H (m)')
axes.set_ylabel('Wave period, T (s)')
axes.legend()
axes.grid()

correl = stats.pearsonr(H, T)
correl_rs = stats.pearsonr(rs_H, rs_T)
print('The correlation between the observations is:', correl[0])
print('The correlation between the simulations is:', correl_rs[0])

