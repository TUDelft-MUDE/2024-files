
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper import plot_contour

data = np.genfromtxt('data.csv', delimiter=";")
data.shape

YOUR_CODE_HERE # probably many lines

def calculate_covariance(X1, X2):
    '''
    Covariance of two random variables X1 and X2 (numpy arrays).
    '''
    YOUR_CODE_HERE # may be more than one line
    return covariance

def pearson_correlation(X1, X2):
    YOUR_CODE_HERE # may be more than one line
    return correl_coeff

covariance = calculate_covariance(data_x1, data_x2)
print(f'The covariance of X1 and X2 is {covariance:.5f}')
correl_coeff = pearson_correlation(data_x1, data_x2)
print(f'The correlation coefficient of X1 and X2 is {correl_coeff:.5f}')

mean_vector = YOUR_CODE_HERE
cov_matrix = YOUR_CODE_HERE
bivar_dist = YOUR_CODE_HERE

plot_contour(YOUR_CODE_HERE, [0, 30, 0, 30], data=YOUR_CODE_HERE);

region_example = np.array([[0, 5, 12, 20, 28, 30],
                           [5, 20, 0, 18, 19, 12]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[23, 13],
             region=region_example);

YOUR_CODE_HERE

YOUR_CODE_HERE

plot_values = np.linspace(-100, 1000, 30)
fig, ax = plt.subplots(1)
ax.hist([YOUR_CODE_HERE, YOUR_CODE_HERE],
         bins=plot_values,
         density=True,
         edgecolor='black');
ax.legend(['Data', 'MCS Sample'])
ax.set_xlabel('$Z(X_1,X_2)$')
ax.set_xlabel('Empirical Density [--]')
ax.set_title('Comparison of Distributions of $Z$');

def ecdf(var):
    x = YOUR_CODE_HERE # sort the values from small to large
    n = YOUR_CODE_HERE # determine the number of datapoints
    y = YOUR_CODE_HERE
    return [y, x]

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

