
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper import plot_contour

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

def get_rho_quadrants(x1, x2):
    mean_x1 = x1.mean()
    mean_x2 = x2.mean()

    upper_right = (x1 > mean_x1) & (x2 > mean_x2)
    upper_left = (x1 <= mean_x1) & (x2 > mean_x2)
    lower_left = (x1 <= mean_x1) & (x2 <= mean_x2)
    lower_right = (x1 > mean_x1) & (x2 <= mean_x2)

    rho_ur = pearson_correlation(x1[upper_right], x2[upper_right])
    rho_ul = pearson_correlation(x1[upper_left], x2[upper_left])
    rho_ll = pearson_correlation(x1[lower_left], x2[lower_left])
    rho_lr = pearson_correlation(x1[lower_right], x2[lower_right])

    return (np.array([rho_ur, rho_ul, rho_ll, rho_lr]),
            ['upper right', 'upper left',
             'lower left', 'lower right'])

def print_rho_quadrants(rho, quadrants):
    for i in range(len(rho)):
        print(f'{quadrants[i]:11s}: {rho[i]:8.5f}')
    
def plot_both_data_sets(x1, x2, s1, s2, savefig=None, showfig=None):
    fig, ax = plt.subplots()
    ax.scatter(x1, x2, label='Data',
            edgecolor='black', linewidth=1, facecolors='none', alpha=0.5)
    ax.scatter(s1, s2, label='Sample from $F_{X_1,X_2}(x_1,x_2)$',
            edgecolor='red', linewidth=1, facecolors='none', alpha=0.4)
    ax.vlines(x1.mean(), 0, 30, label='Mean of X1',
            color='black', linestyle='--', linewidth=1)
    ax.hlines(x2.mean(), 0, 30, label='Mean of X2',
            color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Data and Samples from Bivariate Distribution')
    plt.legend()
    if savefig:
        plt.savefig(savefig)
    if showfig:
        plt.show()
    return fig, ax

current_dir = os.path.dirname(__file__)
file_path = os.path.join(os.path.dirname(__file__),
                         'data.csv')

data = np.genfromtxt(file_path, delimiter=";")

data_x1 = np.array(data[:,0])
data_x2 = np.array(data[:,1])
N = len(data_x1)

X1 = st.norm(data_x1.mean(), data_x1.std())
X2 = st.norm(data_x2.mean(), data_x2.std())
print(data_x1.mean(), data_x1.std())
print(data_x2.mean(), data_x2.std())

cov_x12 = calculate_covariance(data_x1, data_x2)
print(f'The covariance of X1 and X2 is {cov_x12:.5f}')
rho_x12 = pearson_correlation(data_x1, data_x2)
print(f'The correlation coefficient of X1 and X2 is {rho_x12:.5f}')

mean_vector = [data_x1.mean(), data_x2.mean()]
cov_matrix = [[data_x1.std()**2, cov_x12],
              [cov_x12, data_x2.std()**2]]
bivar_dist = st.multivariate_normal(mean_vector, cov_matrix)


print(f'Number of samples: {N}')
samples = bivar_dist.rvs(N)
sample_x1 = samples[:,0]
sample_x2 = samples[:,1]

cov_s12 = calculate_covariance(sample_x1, sample_x2)
print(f'The covariance of the samples is {covariance:.5f}')
rho_s12 = pearson_correlation(sample_x1, sample_x2)
print(f'The correlation coefficient of samples is {rho_s12:.5f}')

plot_both_data_sets(data_x1, data_x2,
                    sample_x1, sample_x2)
                    # savefig=os.path.join(current_dir,'both_datasets.svg'))

rho_quadrants_data, quadrant_names = get_rho_quadrants(data_x1, data_x2)
rho_quadrants_samples, _ = get_rho_quadrants(sample_x1, sample_x2)
print('\nCorrelation Coefficients, Data:')
print_rho_quadrants(rho_quadrants_data, quadrant_names)
print(f'{"all data":11s}: {rho_x12:.3f}')
print('\nCorrelation Coefficients, Samples:')
print_rho_quadrants(rho_quadrants_samples, quadrant_names)
print(f'{"all samples":11s}: {rho_s12:.3f}')