# ---

# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"

# %% [markdown] id="1db6fea9-f3ad-44bc-a4c8-7b2b3008e945"

# %% id="4fc6e87d-c66e-43df-a937-e969acc409f8"
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import probplot

import ipywidgets as widgets
from ipywidgets import interact

plt.rcParams.update({'font.size': 14})

# %% [markdown]

# %% [markdown]

# %% [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"

# %% [markdown]

# %%
def moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S,n):
    """Compute Taylor series approximation of mean and std of V.
    
    Take moments and function parameters as inputs (type float).
    Returns mean and standard deviation of V (type float).
    """
    
    constant = 1/n
    
    dVdR = YOUR_CODE_HERE
    dVdS = YOUR_CODE_HERE
    
    dVdR_2 = YOUR_CODE_HERE
    dVdS_2 = YOUR_CODE_HERE
    
    mu_V_0 = YOUR_CODE_HERE
    mu_V = YOUR_CODE_HERE
    
    var_V = YOUR_CODE_HERE
    sigma_V = YOUR_CODE_HERE
    
    return mu_V, sigma_V

# %% [markdown]

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} id="55ff8dd6-86ef-401a-9a56-02551c348698" outputId="3add4ee9-1054-4726-dc4f-72dca5c1c6c8"
n = 0.013
mu_R = 0.5
mu_S = 0.015
sigma_R = np.linspace(0.0, 0.1, 50)

sigma_S_1 = 0.002
mu_V_1, sigma_V_1 = moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S_1, n)

sigma_S_2 = 0
mu_V_2, sigma_V_2 = moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S_2, n)

fig, ax = plt.subplots(1, 2, figsize = (16, 6))

ax[0].plot(sigma_R, sigma_V_1, linewidth=4)
ax[0].set_ylabel(r'$\sigma_V$ [$m/s$]', size = 20)
ax[0].set_xlabel(r'$\sigma_R$ [$m$]', size = 20)
ax[0].set_title(r'$\sigma_S$ = ' + f'{sigma_S_1} $m/m$, Case 1')
ax[0].set_xlim(0, 0.1)
ax[0].set_ylim(0, 1)

ax[1].plot(sigma_R, sigma_V_2, linewidth=4)
ax[1].set_ylabel(r'$\sigma_V$ [$m/s$]', size = 20)
ax[1].set_xlabel(r'$\sigma_R$ [m]', size = 20)
ax[1].set_title(r'$\sigma_S$ = ' + f'{sigma_S_2} $m/m$, Case 2')
ax[1].set_xlim(0, 0.1)
ax[1].set_ylim(0, 1)
plt.show()

# %% [markdown]

# %% [markdown]

# %% [markdown] id="a7e4c13f-a2ca-4c2d-a3e2-92d4630715a0"

# %% [markdown]

# %%
def function_of_random_variables(R, S):
    V = YOUR_CODE_HERE
    return V

def get_samples(N, sigma_R, mu_R=0.5, mu_S=0.015, sigma_S=0.002, n=0.013):
    """Generate random samples for V from R and S."""
    R = np.random.normal(mu_R, sigma_R, N)
    S = np.random.normal(mu_S, sigma_S, N)
    V = YOUR_CODE_HERE
    return V

V_samples = get_samples(10000, 0.05)

mu_V_samples = YOUR_CODE_HERE
sigma_V_samples = YOUR_CODE_HERE

print('Moments of the SAMPLES:')
print(f'  {mu_V_samples:.4f} m/s is the mean, and')
print(f'  {sigma_V_samples:.4f} m/s is the std dev.')

mu_V_taylor, sigma_V_taylor = moments_of_taylor_approximation(mu_R, mu_S, 0.05, 0.002, n)
print('\nMoments of the TAYLOR SERIES APPROXIMATION:')
print(f'  {mu_V_taylor:.4f} m/s is the mean, and')
print(f'  {sigma_V_taylor:.4f} m/s is the std dev.')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% colab={"base_uri": "https://localhost:8080/", "height": 475, "referenced_widgets": ["b560714d739d431d85b3ca1a9b378c8f", "56b7808a3e2241679b15d517565eaf85", "d867da2ab3d441599b8356ac8e493611", "481c67caa6d1405ea2e00cfe6dbfa32f", "392504e006074b76af62e617c4cde70e", "b0d26f90109f4e0eb6839f0ba43ba980", "ea4c3dc473df41a684cfe7fd1e7fb35d"]} id="80005a5a-510b-4236-a2d6-184d9569eed4" outputId="80ae9e8d-e450-4e17-f092-fbf09fc885e6"
def validate_distribution(N, sigma_R, mu_R=0.5, mu_S=0.015, sigma_S=0.002, n=0.013):
    """Generate samples and plots for V
    
    Compares the sampled distribution of V to a Normal distribution defined
    by the first moments of the error propagation law.
    
    Comparison is made via two plots:
      1. PDF of V~N(mu,sigma) (the approximation) and a histogram (sample)
      2. Probability plot, compares quantiles of sample and CDF of V
    
    Only a plot is returned.
    
    MUDE students fill in the missing code (see: YOUR_CODE_HERE):
      1. Generate samples and find moments
      2. Find moments of the function of random variables using Taylor series
      3. Enter data for the histogram
      4. Define the moments of the Normal distribution to be plotted
      5. Identify the appropriate variables to be printed in the plot titles
      6. Enter the data required for the probability plot
    """
    
    
    V_samples = YOUR_CODE_HERE
    mu_V_samples = YOUR_CODE_HERE
    sigma_V_samples = YOUR_CODE_HERE
    
    
    mu_V_taylor, sigma_V_taylor = YOUR_CODE_HERE

    
    
    xmin = 0
    xmax = 10
    x = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots(1, 2, figsize = (16, 6))
    
    ax[0].hist(YOUR_CODE_HERE, bins = 40, density = True, 
               label = 'Empirical PDF of V')
    
    
    ax[0].plot(x, norm.pdf(x, YOUR_CODE_HERE, YOUR_CODE_HERE), color = 'black',
               lw = 2.5, label='Normal PDF')

    ax[0].legend()
    ax[0].set_xlabel('V [$m/s$]')
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Density')
    ax[0].set_title(f'Simulation with {N} simulated realizations'
                    + '\n' + f'mean = {round(YOUR_CODE_HERE, 3)}' 
                    f'm/s and std = {round(YOUR_CODE_HERE, 3)} m/s')
    
    
    probplot(YOUR_CODE_HERE, dist = norm, fit = True, plot = ax[1])

    ax[1].legend(['Generated samples', 'Normal fit'])
    ax[1].get_lines()[1].set_linewidth(2.5)
    plt.show()

validate_distribution(10000, 0.01)

# %% [markdown]

# %% [markdown]

# %% colab={"base_uri": "https://localhost:8080/", "height": 475, "referenced_widgets": ["b560714d739d431d85b3ca1a9b378c8f", "56b7808a3e2241679b15d517565eaf85", "d867da2ab3d441599b8356ac8e493611", "481c67caa6d1405ea2e00cfe6dbfa32f", "392504e006074b76af62e617c4cde70e", "b0d26f90109f4e0eb6839f0ba43ba980", "ea4c3dc473df41a684cfe7fd1e7fb35d"]} id="80005a5a-510b-4236-a2d6-184d9569eed4" outputId="80ae9e8d-e450-4e17-f092-fbf09fc885e6"
@interact(sigma_R=(0, 0.1, 0.005))
def samples_slideplot(sigma_R):
    validate_distribution(50000, sigma_R);

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
p = YOUR_CODE_HERE

print(f'The probability is {p:0.6e}')

# %% [markdown]

# %% [markdown]

