import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import chi2
from scipy.signal import periodogram

data = np.loadtxt('atm_data.txt', delimiter=',')
time = data[:, 0]
data = data[:, 1]

dt = YOUR_CODE_HERE # Time step
fs = YOUR_CODE_HERE # Sampling frequency

plt.figure(figsize=(10, 3))
plt.plot(time, data)
plt.xlabel('Time [days]')
plt.ylabel('Atmospheric Pressure [hPa]')
plt.title('2 year of atmospheric pressure data')
plt.grid(True)

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
        plt.ylabel('Atmospheric Pressure [hPa]')
        plt.title('Data vs Estimated data')
        plt.grid(True)
        plt.legend()
        plt.subplot(212)
        plt.plot(time, e_hat, label='Residuals')
        plt.xlabel('Time [days]')
        plt.ylabel('Atmospheric Pressure [hPa]')
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
    # Detrending the data
    _, _, e_hat= fit_model(data, time, A)

    N = len(data)

    # Finding the dominant frequency in e_hat
    freqs, pxx = periodogram(YOUR_CODE_HERE, fs=YOUR_CODE_HERE, window='boxcar',
                             nfft=N, return_onesided=False,
                             scaling='density')

    # finding the dominant frequency and amplitude
    # Note: there are many ways to do this
    amplitude = YOUR_CODE_HERE # Amplitude of the dominant frequency
    dominant_frequency = YOUR_CODE_HERE # Dominant frequency

    # Plotting the PSD
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

A = YOUR_CODE_HERE # A-matrix for linear trend (intercept and slope)
dom_f = find_frequency(YOUR_CODE_HERE)
print(f'Dominant Frequency: {YOUR_CODE_HERE} [YOUR_CODE_HERE]')

YOUR_CODE_HERE # may be more than one line or more than one cell

def rewrite_seasonal_comp(a_i, b_i):
    '''
    Function to rewrite the seasonal component in terms of sin and cos
    a_i: seasonal component coefficient for cos
    b_i: seasonal component coefficient for sin
    '''
    A_i = YOUR_CODE_HERE
    theta_i = YOUR_CODE_HERE
    return A_i, theta_i

A = YOUR_CODE_HERE

x_hat, y_hat, e_hat = YOUR_CODE_HERE

a_i = YOUR_CODE_HERE # all the a_i coefficients
b_i = YOUR_CODE_HERE # all the b_i coefficients
freqs = YOUR_CODE_HERE # all the frequencies

assert len(a_i) == len(b_i) == len(freqs), 'The number of coefficients do not match'

print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.2f}')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i = i + 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(YOUR_CODE_HERE, ax=ax, lags=20);
ax.set_xlabel('Lags [days]')
ax.grid()

def AR1(s, time, plot=True):
    '''
    Function to find the AR(1) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    y = YOUR_CODE_HERE
    y_lag_1 = YOUR_CODE_HERE
    # np.atleast_2d is used to convert the 1D array to 2D array,
    # as the fit_model function requires 2D array
    A = np.atleast_2d(y_lag_1).T 

    x_hat, y_hat, e_hat = fit_model(y, time, A)
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(time, y, label='Original Residuals')
        plt.plot(time, y_hat, label='Estimated Residuals')
        plt.xlabel('Time [days]')
        plt.ylabel('Atmospheric Pressure [hPa]')
        plt.title('Original Data vs Estimated Data')
        # plt.xlim([0, 100]) # uncomment this line to zoom in, for better visualization
        plt.grid(True)
        plt.legend()

    print(f'Estimated Parameters:')
    print(f'phi = {x_hat[0]:.4f}')

    return x_hat, e_hat

_, e_hat2 = AR1(YOUR_CODE_HERE)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(YOUR_CODE_HERE, ax=ax, lags=20);
ax.grid()

A_final = YOUR_CODE_HERE # A-matrix for the combined model
x_hat, y_hat, e_hat_final = fit_model(YOUR_CODE_HERE)

N = YOUR_CODE_HERE # Number of data points
p = YOUR_CODE_HERE # Number of parameters
sigma2 = YOUR_CODE_HERE # estimated variance of the residuals
Cov = YOUR_CODE_HERE # Covariance matrix of the parameters
se = np.sqrt(np.diag(Cov)) # Standard errors of the parameters

a_i = YOUR_CODE_HERE # all the a_i coefficients
b_i = YOUR_CODE_HERE # all the b_i coefficients
freqs = YOUR_CODE_HERE # all the frequencies

assert len(a_i) == len(b_i) == len(freqs), 'The number of coefficients do not match'

print(f'Estimated Parameters (standard deviation):')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}\t\t ({se[i]:.3f})')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i = i + 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')

