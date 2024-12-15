# ---

# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"

# %% [markdown]

# %% [markdown]

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper import plot_contour

# %% [markdown]

# %% [markdown]

# %%
data = np.genfromtxt('data.csv', delimiter=";")
data.shape

# %%

data_x1 = np.array(data[:,0])
data_x2 = np.array(data[:,1])

X1 = st.norm(data_x1.mean(), data_x1.std())
X2 = st.norm(data_x2.mean(), data_x2.std())
print(data_x1.mean(), data_x1.std())
print(data_x2.mean(), data_x2.std())

# %% [markdown]

# %% [markdown]

# %%

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

# %%
covariance = calculate_covariance(data_x1, data_x2)
print(f'The covariance of X1 and X2 is {covariance:.5f}')
correl_coeff = pearson_correlation(data_x1, data_x2)
print(f'The correlation coefficient of X1 and X2 is {correl_coeff:.5f}')

# %% [markdown]

# %% [markdown] id="0491cc69"

# %%

# %%

mean_vector = [data_x1.mean(), data_x2.mean()]
cov_matrix = [[data_x1.std()**2, covariance],
              [covariance, data_x2.std()**2]]
bivar_dist = st.multivariate_normal(mean_vector, cov_matrix)

# %%
print(mean_vector, cov_matrix)

# %% [markdown]

# %%
bivar_dist2 = st.multivariate_normal.fit(np.array([data_x1, data_x2]).T)
print(bivar_dist2)

# %%

plot_contour(bivar_dist, [0, 30, 0, 30], data=data);

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown] id="0491cc69"

# %%
region_example = np.array([[0, 5, 12, 20, 28, 30],
                           [5, 20, 0, 18, 19, 12]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[23, 13],
             region=region_example);

# %% [markdown]

# %%

# %% [markdown]

# %% [markdown]

# %%

# %% [markdown]

# %% [markdown] id="0491cc69"

# %% [markdown]

# %%
lower_left = bivar_dist.cdf([20, 20])
union = 1 - lower_left

left = X1.cdf(20)
bottom = X2.cdf(20)
intersection = 1 - (left + bottom - lower_left)

print(f'     lower left:      {lower_left:.5f}')
print(f'     left side:       {left:.5f}')
print(f'     bottom side:     {bottom:.5f}')
print('=============================')
print(f'Case 1, Union:        {union:.5f}')
print(f'Case 2, Intersection: {intersection:.5f}')

# %% [markdown]

# %%
N = data_x1.size

number_of_points_lower_left = sum((data_x1 < 20)&(data_x2 < 20))
lower_left = number_of_points_lower_left/(N + 1)
union = 1 - lower_left

left = sum(data_x2 < 20)/(N + 1)
bottom = sum(data_x1 < 20)/(N + 1)
intersection = 1 - (left + bottom - lower_left)

print(f'     lower left:      {lower_left:.5f}')
print(f'     left side:       {left:.5f}')
print(f'     bottom side:     {bottom:.5f}')
print('=============================')
print(f'Case 1, Union:        {union:.5f}')
print(f'Case 2, Intersection: {intersection:.5f}')

# %% [markdown]

# %% [markdown]

# %%
region_or = np.array([[0, 20, 20, 30],
                      [20, 20, 0, 0]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[20, 20],
             region=region_or,
             data=data);

region_and = np.array([[20, 20, 30],
                      [30, 20, 20]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[20, 20],
             region=region_and,
             data=data);

# %% [markdown]

# %%
plot_x = np.linspace(0, 30, 200)
plot_y = 40 - plot_x**2/20
plot_xy = np.vstack((plot_x, plot_y))

plot_contour(bivar_dist, [0, 30, 0, 30],
             region=plot_xy,
             data=data);

# %% [markdown]

# %%
sample_N = 100*N
sample = bivar_dist.rvs(size=sample_N)
sample_X1 = sample[:,0]
sample_X2 = sample[:,1]

Z = lambda X1, X2: 800 - X1**2 - 20*X2
sample_Z = Z(sample_X1, sample_X2)
Z_less_than_0 = sum(sample_Z<0)

print(f'The number of samples of Z < 0 is: {Z_less_than_0}')
print(f'This is {Z_less_than_0/sample_N*100:.3f}% of all samples.')
print(f'The MCS probability is {Z_less_than_0/sample_N:.3f}.')

# %% [markdown]

# %%
empirical_Z = Z(data_x1, data_x2)
Z_data_less_than_0 = sum(empirical_Z<0)

print(f'The number of data where Z < 0 is: {Z_data_less_than_0}')
print(f'This is {Z_data_less_than_0/(N + 1)*100:.3f}% of all samples.')
print(f'The empirical probability is {Z_data_less_than_0/(N + 1):.3f}.')

# %% [markdown]

# %%
def get_p_and_c_o_v(sample):
    """For sample N return p and c.o.v. for each N_i in N.
    
    Allows one to see how values change as the sample size increases.
    """
    N = len(sample)
    sample_X1 = sample[:,0]
    sample_X2 = sample[:,1]
    Z = lambda X1, X2: 800 - X1**2 - 20*X2
    sample_Z = Z(sample_X1, sample_X2)

    p = np.zeros(N)
    c_o_v = np.zeros(N)
    for i in range(N):
        p[i] = sum(sample_Z[0:i]<0)/(i+1)
        if p[i] == 0:
            c_o_v[i] = 0
        else:
            c_o_v[i] = 1/np.sqrt((i+1)*p[i])

    return p, c_o_v

try_N = 10000
p, c_o_v = get_p_and_c_o_v(bivar_dist.rvs(size=try_N))

# %%
fig, ax1 = plt.subplots()

ax1.plot(range(try_N), p, 'b-')
ax1.set_xlabel('Sample size')
ax1.set_ylabel('Probability', color='b')
ax1.set_title('Probability and c.o.v. for N_i in MCS N')
ax1.set_xlim(100, try_N)
ax1.set_xscale('log')
ax1.set_ylim(1/try_N, 1)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(range(try_N), c_o_v, 'r-')
ax2.set_ylabel('Coefficient of Variation', color='r')
ax2.set_ylim(1e-2, 1)
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor='r')

plt.show()

# %% [markdown]

# %% [markdown] id="0491cc69"

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

plot_values = np.linspace(-100, 1000, 30)
fig, ax = plt.subplots(1)
ax.hist([empirical_Z, sample_Z],
         bins=plot_values,
         density=True,
         edgecolor='black');
ax.legend(['Data', 'MCS Sample'])
ax.set_xlabel('$Z(X_1,X_2)$')
ax.set_xlabel('Empirical Density [--]')
ax.set_title('Comparison of Distributions of $Z$');

# %% [markdown]

# %%

def ecdf(var):
    x = np.sort(var) 
    n = x.size 
    y = np.arange(1, n+1) / (n + 1)
    return [y, x]

# %% [markdown]

# %%

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

# %% [markdown]

# %%

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

# %% [markdown] id="0491cc69"

# %% [markdown]

