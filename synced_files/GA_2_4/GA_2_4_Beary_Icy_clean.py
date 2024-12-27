import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import periodogram
YOUR_CODE_HERE
data = YOUR_CODE_HERE # Temperature data
time_days = YOUR_CODE_HERE # Time in days
def fit_model(data, time, A, plot=False):
    '''
    Function to find the least squares solution of the data
    data: input data
    time: time vector
    A: A-matrix to fit the data
    plot: boolean to plot the results or not
    '''
    x_hat = YOUR_CODE_HERE # least squares solution
    y_hat = YOUR_CODE_HERE # model prediction
    e_hat = YOUR_CODE_HERE # residuals
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(time, data, label='Data')
        plt.plot(time, y_hat, label='Estimated data')
        plt.xlabel('Time [days]')
        plt.ylabel('Temperature [°C]')
        plt.title('Data vs Estimated data')
        plt.grid(True)
        plt.legend()
        plt.subplot(212)
        plt.plot(time, e_hat, label='Residuals')
        plt.xlabel('Time [days]')
        plt.ylabel('Temperature [°C]')
        plt.title('Residuals')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    return x_hat, y_hat, e_hat
def find_frequency(data, time, A, fs, plot=True):
    '''
    Function to find the dominant frequency of the signal
    data: input data
    time: time vector
    A: A-matrix to detrend the data (prior to spectral analysis)
    fs: sampling frequency
    plot: boolean to plot the psd or not
    '''
    _, _, e_hat= fit_model(YOUR_CODE_HERE)
    N = len(data)
    freqs, pxx = periodogram(YOUR_CODE_HERE, fs=YOUR_CODE_HERE, window='boxcar',
                                nfft=N, return_onesided=False,
                                scaling='density')
    amplitude = YOUR_CODE_HERE # Amplitude of the dominant frequency
    dominant_frequency = YOUR_CODE_HERE # Dominant frequency
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(time, e_hat)
        plt.title('Residuals')
        plt.ylabel('Atmospheric Pressure [hPa]')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(freqs[freqs>0], pxx[freqs>0], label='PSD of residuals')
        plt.xlabel('Frequency')
        plt.ylabel('PSD')
        plt.title('Power Spectral Density')
        plt.grid(True)
        plt.plot(dominant_frequency, amplitude, 'ro', label='Dominant Frequency')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
    return dominant_frequency
def rewrite_seasonal_comp(ak, bk):
    '''
    Function to rewrite the seasonal component in terms of sin and cos
    ak: seasonal component coefficient for cos
    bk: seasonal component coefficient for sin
    returns: Ak, theta_k
    '''
    YOUR_CODE_HERE
A = YOUR_CODE_HERE
x_hat, y_hat, e_hat = YOUR_CODE_HERE
plt.figure(figsize=(10, 3))
plt.plot(time_days, data, label='Original data')
plt.plot(time_days, y_hat, label='Estimated trend')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
plt.legend()
plt.figure(figsize=(10, 3))
plt.plot(time_days, e_hat0)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Residuals')
plt.grid(True)
a_i = YOUR_CODE_HERE
b_i = YOUR_CODE_HERE
freqs = YOUR_CODE_HERE
print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}')
print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')
def A1_matrix(A0, break_point):
    '''
    Function to create the A1 matrix
    A0: A matrix under H0
    break_point: break point location
    return: A1 matrix
    '''
    YOUR_CODE_HERE
    return YOUR_CODE_HERE
def LR(e0, e1, cv=100, verbose=True):
    '''
    Function to perform the LR test
    e0: residuals under H0
    e1: residuals under H1
    cv: critical value
    '''
    n = YOUR_CODE_HERE
    SSR0 = YOUR_CODE_HERE
    SSR1 = YOUR_CODE_HERE
    test_stat = YOUR_CODE_HERE
    if test_stat > cv:
        if verbose:
            print(f'Test Statistic: {test_stat:.3f} > Critical Value: {cv:.3f}')
            print('Reject the null hypothesis')
    else:
        if verbose:
            print(f'Test Statistic: {test_stat:.3f} < Critical Value: {cv:.3f}')
            print('Fail to reject the null hypothesis')
    return test_stat
def jump_detection(data, time, A, cv=100, plot=True):
    '''
    Function to detect the jump in the data
    data: input data
    time: time vector
    A: A matrix under H0
    cv: critical value
    plot: boolean to plot the results or not
    '''
    results = YOUR_CODE_HERE
    YOUR_CODE_HERE
    for i in range(1, len(data)):
        A1 = YOUR_CODE_HERE
        if np.linalg.matrix_rank(A1) < A1.shape[1]:
            pass
        else:
            _, _, e_hat1 = YOUR_CODE_HERE
            test_stat = YOUR_CODE_HERE
            results[i] = YOUR_CODE_HERE
    results = np.array(results)
    location = YOUR_CODE_HERE
    value = YOUR_CODE_HERE
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(time, results)
        plt.plot(time[location], value, 'ro', label='offset')
        plt.plot([0, max(time)], [cv, cv], 'k--', label='Critical Value')
        plt.xlabel('Time [days]')
        plt.ylabel('Test Statistic')
        plt.title('LR Test')
        plt.grid(True)
        plt.legend()
    return location, value
YOUR_CODE_HERE
A2 = YOUR_CODE_HERE
x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)
plt.figure(figsize=(10, 3))
plt.plot(time_days, data, label='Original data')
plt.plot(time_days, y_hat, label='Estimated trend')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
plt.legend()
plt.figure(figsize=(10, 3))
plt.plot(time_days, e_hat)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Residuals')
plt.grid(True)
a_i = YOUR_CODE_HERE
b_i = YOUR_CODE_HERE
freqs = YOUR_CODE_HERE
print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}')
print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat, ax=ax, lags=20);
ax.grid()
def AR1(s, time, plot=True):
    '''
    Function to find the AR(1) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    y = YOUR_CODE_HERE
    y_lag_1 = YOUR_CODE_HERE
    A = np.atleast_2d(y_lag_1).T
    x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(time[1:], y, label='Original Residuals')
        ax[0].plot(time[1:], y_hat, label='Estimated Residuals')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Temperature [°C]')
        ax[0].set_title('Original Data vs Estimated Data')
        ax[0].grid(True)
        ax[0].legend()
        plot_acf(e_hat, ax=ax[1], lags=20)
        ax[1].grid()
        fig.tight_layout()
    print(f'Estimated Parameters:')
    print(f'phi = {x_hat[0]:.4f}')
    return x_hat, e_hat
phi_hat_ar1, e_hat_ar1 = AR1(YOUR_CODE_HERE)
def AR2(s, time, plot=True):
    '''
    Function to find the AR(2) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    y = YOUR_CODE_HERE
    y_lag_1 = YOUR_CODE_HERE
    y_lag_2 = YOUR_CODE_HERE
    A = YOUR_CODE_HERE
    x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(time[2:], y, label='Original Residuals')
        ax[0].plot(time[2:], y_hat, label='Estimated Residuals')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Temperature [°C]')
        ax[0].set_title('Original Data vs Estimated Data')
        ax[0].grid(True)
        ax[0].legend()
        plot_acf(e_hat, ax=ax[1], lags=20)
        ax[1].grid()
        fig.tight_layout()
    print(f'Estimated Parameters:')
    print(f'phi_1 = {x_hat[0]:.4f}, phi_2 = {x_hat[1]:.4f}')
    return x_hat, e_hat
phi_hat_ar2, e_hat_ar2 = AR2(YOUR_CODE_HERE)
A_final = YOUR_CODE_HERE
x_hat, y_hat, e_hat_final = fit_model(YOUR_CODE_HERE)
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(YOUR_CODE_HERE, ax=ax, lags=20);
ax.grid()
N = YOUR_CODE_HERE
p = YOUR_CODE_HERE
sigma2 = YOUR_CODE_HERE
Cov = YOUR_CODE_HERE
se = YOUR_CODE_HERE
a_i = YOUR_CODE_HERE
b_i = YOUR_CODE_HERE
freqs = YOUR_CODE_HERE
assert len(a_i) == len(b_i) == len(freqs), 'The number of coefficients do not match'
print(f'Estimated Parameters (standard deviation):')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}\t\t ({se[i]:.3f})')
print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')
