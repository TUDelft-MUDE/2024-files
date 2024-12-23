# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

# ----------------------------------------
# Import
h, u = np.genfromtxt('dataset_hu.csv', delimiter=",", unpack=True, skip_header=True)

# plot time series
fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(h,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Water depth, h (m)')
ax[0].grid()

ax[1].plot(u,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Water velocity, u (m/s)')
ax[1].grid()

# ----------------------------------------
# Statistics for h

print(stats.describe(h))

# ----------------------------------------
# Statistics for u

print(stats.describe(u))

# ----------------------------------------
def ecdf(var):
    x = np.sort(var) # sort the values from small to large
    n = x.size # determine the number of datapoints\
    y = np.arange(1, n+1) / (n+1)
    return [y, x]

# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(h, edgecolor='k', linewidth=0.2, 
             color='cornflowerblue', label='Water depth, h (m)', density = True)
axes[0].set_xlabel('Random variable (X)')
axes[0].set_ylabel('density')
axes[0].hist(u, edgecolor='k', linewidth=0.2, alpha = 0.5, 
             color='grey', label='Water velocity, u (m/s)', density = True)
axes[0].set_title('PDF', fontsize=18)
axes[0].grid()
axes[0].legend()

axes[1].step(ecdf(h)[1], ecdf(h)[0], 
             color='cornflowerblue', label='Water depth, h (m)')
axes[1].set_xlabel('Random variable (X)')
axes[1].set_ylabel('${P[X \leq x]}$')
axes[1].step(ecdf(u)[1], ecdf(u)[0], 
             color='grey', label='Water velocity, u (m/s)')
axes[1].set_title('CDF', fontsize=18)
axes[1].legend()
axes[1].grid()

# ----------------------------------------
params_h = stats.norm.fit(h)
params_u = stats.gumbel_r.fit(u)

# ----------------------------------------
#Graphical method

#Logscale

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].step(ecdf(h)[1], 1-ecdf(h)[0], 
             color='k', label='Water depth, h')
axes[0].plot(ecdf(h)[1], 1-stats.norm.cdf(ecdf(h)[1], *params_h),
             '--', color = 'grey', label='Gaussian')
axes[0].set_xlabel('Water depth, h (m)')
axes[0].set_ylabel('${P[X > x]}$')
axes[0].set_title('h', fontsize=18)
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid()

axes[1].step(ecdf(u)[1], 1-ecdf(u)[0], 
             color='k', label='Water velocity, u')
axes[1].plot(ecdf(u)[1], 1-stats.gumbel_r.cdf(ecdf(u)[1], *params_u),
             '--', color = 'grey', label='Gumbel')
axes[1].set_xlabel('Water velocity, u (m/s)')
axes[1].set_ylabel('${P[X > x]}$')
axes[1].set_title('u', fontsize=18)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid()

# ----------------------------------------
# QQplot

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot([trunc(min(h)), ceil(max(h))], [trunc(min(h)), ceil(max(h))], 'k')
axes[0].scatter(ecdf(h)[1], stats.norm.ppf(ecdf(h)[0], *params_h), 
             color='grey', label='Gaussian')
axes[0].set_xlabel('Observed h (m)')
axes[0].set_ylabel('Estimated h (m)')
axes[0].set_title('h', fontsize=18)
axes[0].set_xlim([trunc(min(h)), ceil(max(h))])
axes[0].set_ylim([trunc(min(h)), ceil(max(h))])
axes[0].legend()
axes[0].grid()

axes[1].plot([trunc(min(u)), ceil(max(u))], [trunc(min(u)), ceil(max(u))], 'k')
axes[1].scatter(ecdf(u)[1], stats.gumbel_r.ppf(ecdf(u)[0], *params_u), 
             color='grey', label='Gumbel')
axes[1].set_xlabel('Observed u (m/s)')
axes[1].set_ylabel('Estimated u (m/s)')
axes[1].set_title('u', fontsize=18)
axes[1].set_xlim([trunc(min(u)), ceil(max(u))])
axes[1].set_ylim([trunc(min(u)), ceil(max(u))])
axes[1].legend()
axes[1].grid()

# ----------------------------------------
#KStest

_, p_h = stats.kstest(h,stats.norm.cdf, args=params_h)
_, p_u = stats.kstest(u,stats.gumbel_r.cdf, args=params_u)

print('The p-value for the fitted Gaussian distribution to h is:', round(p_h, 3))
print('The p-value for the fitted Gumbel distribution to u is:', round(p_u, 3))

# ----------------------------------------
# Here, the solution is shown for the Lognormal distribution

# Draw random samples
rs_h = stats.norm.rvs(*params_h, size = 10000)
rs_u = stats.gumbel_r.rvs(*params_u, size = 10000)

#Compute Fh
rs_q = rs_h * rs_u

#repeat for observations
q = h * u

#plot the PDF and the CDF
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
axes[0].hist(rs_q, edgecolor='k', linewidth=0.2, density = True, label = 'From simulations')
axes[0].hist(q, edgecolor='k', facecolor = 'orange', alpha = 0.5, linewidth=0.2, 
             density = True, label = 'From observations')
axes[0].set_xlabel('Discharge (m3/s)')
axes[0].set_ylabel('density')
axes[0].set_title('PDF', fontsize=18)
axes[0].legend()
axes[0].grid()

axes[1].step(ecdf(rs_q)[1], 1-ecdf(rs_q)[0], label = 'From simulations')
axes[1].step(ecdf(q)[1], 1-ecdf(q)[0], color = 'orange', label = 'From observations')
axes[1].set_xlabel('Discharge (m3/s)')
axes[1].set_ylabel('${P[X > x]}$')
axes[1].set_title('Exceedance plot', fontsize=18)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid()

# ----------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_h, rs_u, 40, 'k', label = 'Simulations')
axes.scatter(h, u, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Water depth, h (m)')
axes.set_ylabel('Flow velocity, u (m/s)')
axes.legend()
axes.grid()

# ----------------------------------------
#Correlation
correl = stats.pearsonr(h, u)
correl_rs = stats.pearsonr(rs_h, rs_u)
print('The correlation between the observations is:', correl[0])
print('The correlation between the simulations is:', correl_rs[0])

