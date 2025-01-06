%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from helper import plot_contour
from helper import Bivariate
data_set = 'dataset_traffic'
N_factor_MCS = 100
make_figs = True
save_figs = False
bivariate_lim = [0, 250, -20, 100]
plot_x2 = lambda x1, z90: (z90 - 143*x1)/469
Z = lambda x1, x2: 143*x1 + 469*x2
data_path = "data/dataset_traffic.csv"
data = np.genfromtxt(data_path,
                     delimiter=",",
                     unpack=True,
                     skip_header=True)
data_x1 = np.array(data[0,:])
data_x2 = np.array(data[1,:])
parameters1 = st.uniform.fit_loc_scale(data_x1)
dist_x1 = st.uniform(*parameters1)
parameters2 = st.norm.fit_loc_scale(data_x2)
dist_x2 = st.gumbel_r(*parameters2)
N = len(data_x1)
print(f'shape of data: {data.shape}')
print(f'shape of data_x1: {data_x1.shape}')
print(f'min/max of data_x1: {data_x1.min()}, {data_x1.max()}')
print(f'shape of data_x2: {data_x2.shape}')
print(f'min/max of data_x2: {data_x2.min()}, {data_x2.max()}')
print('\n')
print('mean and std of X1 and X2: ')
print(data_x1.mean(), data_x1.std())
print(data_x2.mean(), data_x2.std())
def calculate_covariance(X1, X2):
    '''
    Covariance of two random variables X1 and X2 (numpy arrays).
    '''
    mean_x1 = X1.mean()
    mean_x2 = X2.mean()
    diff_x1 = X1 - mean_x1
    diff_x2 = X2 - mean_x2
    product = diff_x1 * diff_x2
    covariance = product.mean()
    return covariance
def pearson_correlation(X1, X2):
    covariance = calculate_covariance(X1, X2)
    correl_coeff = covariance/(X1.std()*X2.std())
    return correl_coeff
cov_x12 = calculate_covariance(data_x1, data_x2)
print(f'The covariance of X1 and X2 is {cov_x12:.5f}')
rho_x12 = pearson_correlation(data_x1, data_x2)
print(f'The correlation coefficient of X1 and X2 is {rho_x12:.5f}')
bivar_dist = Bivariate(dist_x1, dist_x2, rho_x12)
plot_contour(bivar_dist, bivariate_lim, data=data)
region_example = np.array([[0, 5, 12, 20, 28, 30],
                           [5, 20, 0, 18, 19, 12]])
plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[23, 13],
             region=region_example);
x90_1 = dist_x1.ppf(.9)
x90_2 = dist_x2.ppf(.9)
region_or = np.array([[bivariate_lim[0], x90_1, x90_1, bivariate_lim[1]],
                      [x90_2, x90_2, bivariate_lim[2], bivariate_lim[2]]])
region_and = np.array([[x90_1, x90_1, bivariate_lim[1]],
                       [bivariate_lim[3], x90_2, x90_2]])
lower_left = bivar_dist.cdf([x90_1, x90_2])
union = 1 - lower_left
left = dist_x1.cdf(x90_1)
bottom = dist_x2.cdf(x90_2)
intersection = 1 - (left + bottom - lower_left)
print(f'Case 1 and 2 probabilities')
print(f'     lower left:      {lower_left:.5f}')
print(f'     left side:       {left:.5f}')
print(f'     bottom side:     {bottom:.5f}')
print('=============================')
print(f'Case 1, Union:        {union:.5f}')
print(f'Case 2, Intersection: {intersection:.5f}')
print('\n')
N = data_x1.size
number_of_points_lower_left = sum((data_x1 < x90_1)&(data_x2 < x90_2))
lower_left = number_of_points_lower_left/(N + 1)
union = 1 - lower_left
left = sum(data_x1 < x90_1)/(N + 1)
bottom = sum(data_x2 < x90_2)/(N + 1)
intersection = 1 - (left + bottom - lower_left)
print(f'Case 1 and 2 empirical probabilities')
print(f'     lower left:      {lower_left:.5f}')
print(f'     left side:       {left:.5f}')
print(f'     bottom side:     {bottom:.5f}')
print('=============================')
print(f'Case 1, Union:        {union:.5f}')
print(f'Case 2, Intersection: {intersection:.5f}')
print('\n')
if make_figs:
    plot_contour(bivar_dist, bivariate_lim, data=data,
                region=region_or, case=[x90_1, x90_2])
    if save_figs:
        plt.savefig(os.path.join(figure_path,'figure_2.svg'))
if make_figs:
    plot_contour(bivar_dist, bivariate_lim, data=data,
                region=region_and, case=[x90_1, x90_2])
    if save_figs:
        plt.savefig(os.path.join(figure_path,'figure_3.svg'))
z_90 = Z(x90_1, x90_2)
plot_Z_x1 = np.linspace(bivariate_lim[0], bivariate_lim[1], 100)
plot_Z_x2 = plot_x2(plot_Z_x1, z_90)
region_Z = np.array([plot_Z_x1, plot_Z_x2])
if make_figs:
    plot_contour(bivar_dist, bivariate_lim, data=data,
                region=region_Z, case=[x90_1, x90_2])
    if save_figs:
        plt.savefig(os.path.join(figure_path,'figure_4.svg'))
sample_N = N_factor_MCS*N
sample = bivar_dist.rvs(size=sample_N)
sample_X1 = sample[:,0]
sample_X2 = sample[:,1]
sample_Z = Z(sample_X1, sample_X2)
Z_beyond_90 = sum(sample_Z>z_90)
p_Z90 = Z_beyond_90/(sample_N + 1)
print(f'Z case MCS')
print(f'N = {sample_N}')
print(f'The number of samples of Z < 0 is: {Z_beyond_90}')
print(f'This is {p_Z90*100:.3f}% of all samples.')
print(f'The MCS probability is {p_Z90:.3f}.')
print(f'The c.o.v. is of p is {1/np.sqrt((sample_N+1)*p_Z90):.3f}.')
print('\n')
empirical_Z = Z(data_x1, data_x2)
Z_data_beyond_90 = sum(empirical_Z>z_90)
p_Z90_data = Z_data_beyond_90/(N + 1)
print(f'Z case empirical')
print(f'N = {N}')
print(f'The number of data where Z < 0 is: {Z_data_beyond_90}')
print(f'This is {p_Z90_data*100:.3f}% of all samples.')
print(f'The empirical probability is {p_Z90_data:.3f}.')
print(f'The c.o.v. is of p is {1/np.sqrt((N+1)*p_Z90_data):.3f}.')
print('\n')
plot_values = np.linspace(sample_Z.min(), sample_Z.max(), 30)
fig, ax = plt.subplots(1)
ax.hist([empirical_Z, sample_Z],
         bins=plot_values,
         density=True,
         edgecolor='black');
ax.legend(['Data', 'MCS Sample'])
ax.set_xlabel('$Z(X_1,X_2)$')
ax.set_xlabel('Empirical Density [--]')
ax.set_title('Comparison of Distributions of $Z$');
def ecdf(var):
    x = np.sort(var) # sort the values from small to large
    n = x.size # determine the number of datapoints
    y = np.arange(1, n+1) / (n + 1)
    return [y, x]
fig, axes = plt.subplots(1, 1, figsize=(8, 5))
axes.step(ecdf(empirical_Z)[1], ecdf(empirical_Z)[0], 
          color='k', label='Data')
axes.step(ecdf(sample_Z)[1], ecdf(sample_Z)[0],
          color='r', label='MCS Sample')
axes.set_xlabel('$Z(X_1,X_2)$')
axes.set_ylabel(r'CDF, $\mathrm{P}[Z < z]$')
axes.set_title('Comparison: CDF (log scale expands lower tail)')
axes.set_yscale('log')
axes.legend()
axes.grid()
fig, axes = plt.subplots(1, 1, figsize=(8, 5))
axes.step(ecdf(empirical_Z)[1], 1-ecdf(empirical_Z)[0], 
          color='k', label='Data')
axes.step(ecdf(sample_Z)[1], 1-ecdf(sample_Z)[0],
          color='r', label='MCS Sample')
axes.set_xlabel('$Z(X_1,X_2)$')
axes.set_ylabel(r'Exceedance Probability, $\mathrm{P}[Z > z]$')
axes.set_title('Comparison: CDF (log scale expands upper tail)')
axes.set_yscale('log')
axes.legend()
axes.grid()
