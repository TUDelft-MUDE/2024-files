import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import probplot
plt.rcParams.update({'font.size': 14})
def stefan(constant, H0, Ts, Tfr, time):
    return np.sqrt(constant*time*abs(Ts-Tfr) + H0**2)
print('Ice thickness: ' +
      f'{stefan(1.44*10**(-8), 0.15, 261, 273, 3*24*3600):.3f} m')
def H_taylor(mu_H0, mu_iT, sigma_H0, sigma_iT):
    """ Taylor series approximation of mean and std of H"""
    constant = 1.44*10**(-8)
    time = 3*24*3600
    dhdiT = ((constant*time*mu_iT + mu_H0**2)**(-0.5))*constant/2*time
    dhdH0 = ((constant*time*mu_iT + mu_H0**2)**(-0.5))*mu_H0
    dhdiT_2 = -((constant/2*time)**2)*(constant*time*mu_iT+mu_H0**2)**(-1.5)
    dhdH0_2 = (((constant*time*mu_iT + mu_H0**2)**(-0.5)) - 
              mu_H0**2*(constant*time*mu_iT + mu_H0**2)**(-1.5))
    mu_H_0 = np.sqrt(constant*time*mu_iT + mu_H0**2)
    mu_H = mu_H_0 + 0.5*dhdiT_2*sigma_iT**2 + 0.5*dhdH0_2*sigma_H0**2
    var_H = (dhdiT**2)*sigma_iT**2 + (dhdH0**2)*sigma_H0**2
    sigma_H = np.sqrt(var_H)
    return mu_H, sigma_H
def samples_plot(N, mu_H0, mu_iT, sigma_H0, sigma_iT):
    """Generate samples and plots for V
    Compares the approximated Normal distribution of V to numerically
    approximated distribution, found by sampling from the input
    distributions.
    Return: a plot and the mean and std dev of simulated values of H_ice.
    """
    H0_samples = np.random.normal(mu_H0, sigma_H0, N)
    iT_samples = np.random.normal(mu_iT, sigma_iT, N)
    count_negative_iT = sum(iT_samples < 0)
    if count_negative_iT > 0:
        iT_samples[iT_samples < 0] = 0
        print(f'Number of iT samples adjusted to 0: {count_negative_iT} '+
              f'({count_negative_iT/N*100:.1f}% of N)')
    constant = 1.44*10**(-8)
    time = 3*24*3600
    h_samples = np.sqrt(constant*time*iT_samples + H0_samples**2)
    mu_H = np.mean(h_samples)
    sigma_H = np.std(h_samples)
    xmin = 0.0
    xmax = 0.5
    x = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots(1, 2, figsize = (16, 6))
    ax[0].hist(h_samples,
               bins = 40, density = True,
               edgecolor='black', linewidth=1.2, 
               label = 'Empirical PDF of ${H_{ice}}$')
    mu_H_taylor, sigma_H_taylor = H_taylor(mu_H0, mu_iT, sigma_H0, sigma_iT)
    ax[0].plot(x, norm.pdf(x, mu_H_taylor, sigma_H_taylor), color = 'black',
               lw = 2.5, label='Normal PDF')
    ax[0].set_xlim(xmin, xmax)
    ax[0].legend()
    ax[0].set_xlabel('${H_{ice} [m]}$')
    ax[0].set_ylabel('Density')
    ax[0].set_title(f'Simulation with {N} simulated realizations'
                    + '\n' + f'mean = {mu_H:.3e}' 
                    f'm and std = {sigma_H:.3e} m')
    probplot(h_samples, dist = norm, fit = True, plot = ax[1])
    ax[1].legend(['Generated samples', 'Normal fit'])
    ax[1].get_lines()[1].set_linewidth(2.5)
    plt.show()
    return mu_H, sigma_H, h_samples
mu_iT = 10
sigma_iT = 4
mu_H0 = 0.20
sigma_H0 = 0.03
N = 10000
mu_H, sigma_H = H_taylor(mu_H0, mu_iT, sigma_H0, sigma_iT)
print('Comparison of propagated and simulated distributions:\n')
mu_H_simulated, sigma_H_simulated, _ = samples_plot(N,
                                                    mu_H0, mu_iT,
                                                    sigma_H0, sigma_iT)
print('\n\nMean and standard deviation of linearized function:')
print('  \N{GREEK SMALL LETTER MU}',
        '\N{LATIN SUBSCRIPT SMALL LETTER H}=',
      f'{mu_H:.3f}', 'm')
print('  \N{GREEK SMALL LETTER SIGMA}',
        '\N{LATIN SUBSCRIPT SMALL LETTER H}=',
      f'{sigma_H:.3f}', 'm')
print('\n\nMean and standard deviation of simulated distribution:')
print('  \N{GREEK SMALL LETTER MU}',
        '\N{LATIN SUBSCRIPT SMALL LETTER H} =',
      f'{mu_H_simulated:.3f}', 'm')
print('  \N{GREEK SMALL LETTER SIGMA}',
        '\N{LATIN SUBSCRIPT SMALL LETTER H}=',
      f'{sigma_H_simulated:.3f}', 'm')
print('\n')
for N in [5, 50, 500, 5000, 50000]:
    mu_H_simulated, sigma_H_simulated, h_samp = samples_plot(N,
                                                             mu_H0,
                                                             mu_iT,
                                                             sigma_H0,
                                                             sigma_iT)
    print(f'For N = {N} samples:')
    print(f'    mean = {mu_H_simulated:.3f} m')
    print(f'    std = {sigma_H_simulated:.3f} m\n')
for i in np.linspace(0.1, 0.4, 10):
    print(f'for an ice thickness of {i:5.2f} m --> ' +
          f'{100*sum(h_samp <= i)/len(h_samp):8.4f}% of samples, ' +
          f'{100*norm.cdf(i, mu_H, sigma_H):8.4f}% of distribution')
