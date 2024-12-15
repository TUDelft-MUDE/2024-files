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

# %% [markdown]

# %% [markdown]

# %%
def moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S,n):
    """Compute Taylor series approximation of mean and std of V.
    
    Take moments and function parameters as inputs (type float).
    Returns mean and standard deviation of V (type float).
    """
    
    constant = 1/n
    
    
    
    
    dVdR = (2/3)*constant*(mu_R**(-1/3))*(mu_S**(1/2))
    dVdS = (1/2)*constant*(mu_R**(2/3))*(mu_S**(-1/2))
    
    
    
    
    dVdR_2 = (-2/9)*constant*(mu_R**(-4/3))*(mu_S**(1/2))
    dVdS_2 = (-1/4)*constant*(mu_R**(2/3))*(mu_S**(-3/2))
    
    
    
    
    mu_V_0 = constant*(mu_R**(2/3))*(mu_S**(1/2))
    mu_V = mu_V_0 + 0.5*dVdR_2*sigma_R**2 + 0.5*dVdS_2*sigma_S**2
    
    
    
    
    var_V = (dVdR**2)*sigma_R**2 + (dVdS**2)*sigma_S**2
    sigma_V = np.sqrt(var_V)
    
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

# %% [markdown] id="d3bdade1-2694-4ee4-a180-3872ee17a76d"

# %% [markdown] id="a7e4c13f-a2ca-4c2d-a3e2-92d4630715a0"

# %% [markdown]

# %%
def function_of_random_variables(R, S):
    
    
    V = 1/n*R**(2/3)*S**(1/2)
    return V

def get_samples(N, sigma_R, mu_R=0.5, mu_S=0.015, sigma_S=0.002, n=0.013):
    """Generate random samples for V from R and S."""
    R = np.random.normal(mu_R, sigma_R, N)
    S = np.random.normal(mu_S, sigma_S, N)
    
    
    V = function_of_random_variables(R, S)
    return V

V_samples = get_samples(10000, 0.05)

mu_V_samples = V_samples.mean()
sigma_V_samples = V_samples.std()

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
      2. Enter data for the histogram
      3. Define the moments of the Normal distribution to be plotted
      4. Identify the appropriate variables to be printed in the plot titles
      5. Enter the data required for the probability plot
    """
    
    
    
    
    
    
    V_samples = get_samples(N, sigma_R)
    mu_V_samples = V_samples.mean()
    sigma_V_samples = V_samples.std()
    
    
    
    mu_V_taylor, sigma_V_taylor = moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S, n)

    
    
    xmin = 0
    xmax = 10
    x = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots(1, 2, figsize = (16, 6))
    
    
    
    
    ax[0].hist(V_samples, bins = 40, density = True, 
               label = 'Empirical PDF of V')
    
    
    
    
    
    
    ax[0].plot(x, norm.pdf(x, mu_V_taylor, sigma_V_taylor), color = 'black',
               lw = 2.5, label='Normal PDF')
    ax[0].legend()
    ax[0].set_xlabel('V [$m/s$]')
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Density')
    
    
    
    
    
    ax[0].set_title(f'Simulation with {N} simulated realizations'
                    + '\n' + f'mean = {round(mu_V_samples, 3)}' 
                    f'm/s and std = {round(sigma_V_samples, 3)} m/s')
    
    
    
    
    
    probplot(V_samples, dist = norm, fit = True, plot = ax[1])
    ax[1].legend(['Generated samples', 'Normal fit'])
    ax[1].get_lines()[1].set_linewidth(2.5)
    plt.show()

validate_distribution(10000, 0.01)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% colab={"base_uri": "https://localhost:8080/", "height": 475, "referenced_widgets": ["b560714d739d431d85b3ca1a9b378c8f", "56b7808a3e2241679b15d517565eaf85", "d867da2ab3d441599b8356ac8e493611", "481c67caa6d1405ea2e00cfe6dbfa32f", "392504e006074b76af62e617c4cde70e", "b0d26f90109f4e0eb6839f0ba43ba980", "ea4c3dc473df41a684cfe7fd1e7fb35d"]} id="80005a5a-510b-4236-a2d6-184d9569eed4" outputId="80ae9e8d-e450-4e17-f092-fbf09fc885e6"
@interact(sigma_R=(0, 0.1, 0.005))
def samples_slideplot(sigma_R):
    validate_distribution(50000, sigma_R);

# %% [markdown] id="782c842e-ceb8-4e3c-b767-1f3efa4fb9b2"

# %% [markdown]

# %% [markdown]

# %%

p = 2*norm.cdf(-3)

print(f'The probability is {p:0.6e}')

print(f'The probability is {2*norm.cdf(-2.5):0.6e} for sigma_R = 0.01')
print(f'The probability is {2*norm.cdf(-3.0):0.6e} for sigma_R = 0.05')
print(f'The probability is {2*norm.cdf(-3.5):0.6e} for sigma_R = 0.10')

# %% [markdown]

# %% [markdown]

