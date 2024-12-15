import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats.distributions import chi2

plt.rcParams.update({'font.size': 14})

def compute_y(x, times, rain):
    '''Functional model, q: response due to rain event.
    
    Inputs:
      x: tuple, list or array of parameters (d, a, r)
      times: array of times
      rain: tuple, list or array describing rain event
             (rain (m), start day, stop day)

    Returns: ndarray of groundwater levels due to rain event
    '''
    h = (x[0]
         + (np.heaviside(times - rain[1], 1)
            *rain[0]*x[2]*(1 - np.exp(-(times - rain[1])/x[1]))
            - np.heaviside(times - rain[2], 1)
              *rain[0]*x[2]*(1 - np.exp(-(times - rain[2])/x[1]))
            )
         )
    return h

d = YOUR_CODE_HERE
a = YOUR_CODE_HERE
r = YOUR_CODE_HERE

test_n_days = 25
test_times = np.arange(1, test_n_days+1, 0.1)
test_h_t = compute_y((d, a, r), test_times, (0.05, 4, 7))
plt.figure(figsize=(10, 6))
plt.plot(test_times, test_h_t,'r-', linewidth=4.0)
plt.xlabel('Time [days]')
plt.ylabel('Water level [m]')
plt.xlim([0, test_n_days])
plt.ylim([0, 5]);

n_days = 25
y = np.genfromtxt('./data/well_levels.csv' , delimiter=" ,")
times = np.arange(1, n_days+1, 1)

plt.figure(figsize=(10, 6))
plt.plot(times, y,'co', mec='black')
plt.xlabel('Time [days]')
plt.ylabel('Waterlevel [m]');

def jacobian(x, times, rain):
    '''Compute Jacobian of the functional model.
    
    Input:
      x: tuple, list or array of parameters (d, a, r)
      times: array of times
      rain: tuple, list or array describing rain event
             (rain (m), start day, stop day)

    Outputs: The Jacobian
             (partial derivatives w.r.t. d, a, and r)
    '''

    dqdd = YOUR_CODE_HERE
    dqda = YOUR_CODE_HERE
    dqdr = YOUR_CODE_HERE
    J = YOUR_CODE_HERE
    return J

d_init = YOUR_CODE_HERE
a_init = YOUR_CODE_HERE
r_init = YOUR_CODE_HERE

rain_event = (0.05, 4, 7)

sigma = 0.01
var_Y = sigma**2
inv_Sigma_Y = 1/var_Y * np.eye(len(y))

max_iter = 50
x_norm = 10000  # initialize stop criteria (norm of x)

param_init = np.array([d_init, a_init, r_init])
x_hat_i = np.zeros((3, max_iter))
x_hat_i[:] = np.nan
x_hat_i[:, 0] = param_init

iteration = 0

while x_norm >= 1e-12 and iteration < max_iter - 1:

    y_comp_i = compute_y(x_hat_i[:, iteration], times, rain_event)
    
    Delta_y_i = YOUR_CODE_HERE
    
    J_i = jacobian(x_hat_i[:, iteration], times, rain_event)
    N_i = J_i.T @ inv_Sigma_Y @ J_i
    
    Delta_x_hat_i = YOUR_CODE_HERE
    
    x_hat_i[:, iteration+1] = x_hat_i[:, iteration] + Delta_x_hat_i

    x_norm = YOUR_CODE_HERE

    iteration += 1

    if iteration == max_iter - 1:
        print("Number of iterations too large, check initial values.")

print('Initial estimates:')
print(f'base level [m]:\t\t {round(YOUR_CODE_HERE, 4)}')
print(f'scaling parameter:\t {round(YOUR_CODE_HERE, 4)}')
print(f'response [m/m]:\t\t {round(YOUR_CODE_HERE, 4)}','\n')

print('Final estimates:')
print(f'base level [m]:\t\t {round(YOUR_CODE_HERE, 4)}')
print(f'scaling parameter:\t {round(YOUR_CODE_HERE, 4)}')
print(f'response [m/m]:\t\t {round(YOUR_CODE_HERE, 4)}')

print(f'\nNumber of unknowns:\t {YOUR_CODE_HERE}')
print(f'Number of observations:\t {YOUR_CODE_HERE}')
print(f'Redundancy:\t\t {YOUR_CODE_HERE}')

y_hat = YOUR_CODE_HERE

plt.figure(figsize=(10, 6))
t = np.arange(1, n_days+1, 0.1)
plt.plot(times, y_hat , linewidth=4,
         label='Gauss Newton fit', color='black')
plt.plot(times, y, 'co', mec='black',
         markersize=10, label='Observations')
plt.legend()
plt.xlabel('Time [days]')
plt.ylabel('Water level [m]');

params = ['d', 'a', 'r']
fig, ax = plt.subplots(1,3, figsize=(16,4))
plt.subplots_adjust(wspace=0.35)
for i in range(3):
    ax[i].plot(x_hat_i[i, :].T, linewidth=4)
    ax[i].set_title(f'Convergence of {params[i]}')
    ax[i].set_xlabel(f'Number of iterations')
    ax[i].set_ylabel(f'{params[i]}')

e_hat = y - y_hat
Tq = e_hat.T @ inv_Sigma_Y @ e_hat

alpha = 0.05

q = YOUR_CODE_HERE

k = chi2.ppf(1 - alpha, q)

if YOUR_CODE_HERE
    print(f"(T = {Tq:.1f}) < (K = {k:.1f}), OMT is accepted.")
else:
    print(f"(T = {Tq:.1f}) > (K = {k:.1f}), OMT is rejected.")

