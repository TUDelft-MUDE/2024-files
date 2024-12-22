# ----------------------------------------
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# this will print all float arrays with 3 decimal places
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})



# ----------------------------------------
y = [59.82, 57.20, 59.09, 59.49, 59.68, 59.34, 60.95, 59.34, 55.38, 54.33, 48.71, 48.47]

times = YOUR_CODE_HERE
number_of_observations = YOUR_CODE_HERE
number_of_parameters = YOUR_CODE_HERE
print(f'Dimensions of the design matrix A:')
print(f'  {YOUR_CODE_HERE} rows')
print(f'  {YOUR_CODE_HERE} columns')

# ----------------------------------------
column_1 = np.array([33333, 0, 0])
column_2 = 99999*np.ones(3)

example_1 = np.diagflat([column_1, column_2])
example_2 = np.column_stack((column_1, column_2))

print(example_1, '\n\n', example_2)


# ----------------------------------------
A = YOUR_CODE_HERE
Sigma_Y = YOUR_CODE_HERE

assert A.shape == (number_of_observations, number_of_parameters)

# ----------------------------------------
inv_Sigma_Y = np.linalg.inv(Sigma_Y)

xhat_LS = YOUR_CODE_HERE
xhat_BLU = YOUR_CODE_HERE

print('LS estimates in [m], [m/month], [m], resp.:\t', xhat_LS)
print('BLU estimates in [m], [m/month], [m], resp.:\t', xhat_BLU)

# ----------------------------------------
LT = YOUR_CODE_HERE
Sigma_xhat_LS = LT YOUR_CODE_HERE
std_xhat_LS = YOUR_CODE_HERE

Sigma_xhat_BLU = YOUR_CODE_HERE
std_xhat_BLU = YOUR_CODE_HERE

print(f'Precision of LS  estimates in [m], [m/month], [m], resp.:', std_xhat_LS)
print(f'Precision of BLU estimates in [m], [m/month], [m], resp.:', std_xhat_BLU)

# ----------------------------------------
eTe_LS = (y - A @ xhat_LS).T @ (y - A @ xhat_LS)
eTe_BLU = (y - A @ xhat_BLU).T @ inv_Sigma_Y @ (y - A @ xhat_BLU)

print(f'Weighted squared norm of residuals with LS  estimation: {eTe_LS:.3f}')
print(f'Weighted squared norm of residuals with BLU estimation: {eTe_BLU:.3f}')

plt.figure()
plt.rc('font', size=14)
plt.plot(times, y, 'kx', label='observations')
plt.plot(times, A @ xhat_LS, color='r', label='LS')
plt.plot(times, A @ xhat_BLU, color='b', label='BLU')
plt.xlim(-0.2, (number_of_observations - 1) + 0.2)
plt.xlabel('time [months]')
plt.ylabel('height [meters]')
plt.legend(loc='best');

# ----------------------------------------
yhat_LS = YOUR_CODE_HERE
Sigma_Yhat_LS = YOUR_CODE_HERE
yhat_BLU = YOUR_CODE_HERE
Sigma_Yhat_BLU = YOUR_CODE_HERE

alpha = YOUR_CODE_HERE
k98 = YOUR_CODE_HERE

CI_y = YOUR_CODE_HERE
CI_yhat_LS = YOUR_CODE_HERE
CI_yhat_BLU = YOUR_CODE_HERE

# ----------------------------------------
plt.figure(figsize = (10,4))
plt.rc('font', size=14)
plt.subplot(121)
plt.plot(times, y, 'kx', label='observations')
plt.plot(times, yhat_LS, color='r', label='LS')
plt.plot(times, yhat_LS + CI_yhat_LS, 'r:', label=f'{100*(1-alpha):.1f}% conf.')
plt.plot(times, yhat_LS - CI_yhat_LS, 'r:')
plt.xlabel('time [months]')
plt.ylabel('height [meters]')
plt.legend(loc='best')
plt.subplot(122)
plt.plot(times, y, 'kx', label='observations')
plt.errorbar(times, y, yerr = CI_y, fmt='', capsize=5, linestyle='', color='blue', alpha=0.6)
plt.plot(times, yhat_BLU, color='b', label='BLU')
plt.plot(times, yhat_BLU + CI_yhat_BLU, 'b:', label=f'{100*(1-alpha):.1f}% conf.')
plt.plot(times, yhat_BLU - CI_yhat_BLU, 'b:')
plt.xlim(-0.2, (number_of_observations - 1) + 0.2)
plt.xlabel('time [months]')
plt.legend(loc='best');

# ----------------------------------------
rate =YOUR_CODE_HERE
CI_rate = YOUR_CODE_HERE

amplitude = YOUR_CODE_HERE
CI_amplitude = YOUR_CODE_HERE

print(f'The melting rate is {rate:.3f} 짹 {CI_rate:.3f} m/month (98% confidence level)')
print(f'The amplitude of the annual signal is {amplitude:.3f} 짹 {CI_amplitude:.3f} m (98% confidence level)')

