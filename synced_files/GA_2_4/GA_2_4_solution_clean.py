# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import periodogram

# ----------------------------------------
# YOUR_CODE_HERE
# data = YOUR_CODE_HERE # Temperature data
# time_hours = YOUR_CODE_HERE # Time in hours

# SOLUTION
# Reading the data from the file
data = np.loadtxt('temperature.csv')
time_hours = np.arange(0, len(data))
time_days = time_hours / 24
dt = time_days[1] - time_days[0]
fs = 1 / dt

# Plotting the data
plt.figure(figsize=(10, 3))
plt.plot(time_days, data)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [째C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
# END SOLUTION BLOCK

# ----------------------------------------
def fit_model(data, time, A, plot=False):
    '''
    Function to find the least squares solution of the data
    data: input data
    time: time vector
    A: A-matrix to fit the data
    plot: boolean to plot the results or not
    '''

    # x_hat = YOUR_CODE_HERE # least squares solution
    # y_hat = YOUR_CODE_HERE # model prediction
    # e_hat = YOUR_CODE_HERE # residuals

    # SOLUTION
    x_hat = np.linalg.solve(A.T @ A, A.T @ data)
    y_hat = A @ x_hat
    e_hat = data - y_hat
    # END SOLUTION BLOCK

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(time, data, label='Data')
        plt.plot(time, y_hat, label='Estimated data')
        plt.xlabel('Time [days]')
        plt.ylabel('Temperature [째C]')
        plt.title('Data vs Estimated data')
        plt.grid(True)
        plt.legend()
        plt.subplot(212)
        plt.plot(time, e_hat, label='Residuals')
        plt.xlabel('Time [days]')
        plt.ylabel('Temperature [째C]')
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
    # freqs, pxx = periodogram(YOUR_CODE_HERE, fs=YOUR_CODE_HERE, window='boxcar',
    #                          nfft=N, return_onesided=False,
    #                          scaling='density')
    
    # SOLUTION
    # Finding the dominant frequency in e_hat
    freqs, pxx = periodogram(e_hat, fs=fs, window='boxcar',
                                nfft=N, return_onesided=False,
                                scaling='density')
    # END SOLUTION BLOCK

    # finding the dominant frequency and amplitude
    # Note: there are many ways to do this
    # amplitude = YOUR_CODE_HERE # Amplitude of the dominant frequency
    # dominant_frequency = YOUR_CODE_HERE # Dominant frequency

    # SOLUTION
    # finding the dominant frequency and amplitude
    dominant_frequency, amplitude = freqs[np.argmax(pxx)], np.max(pxx)
    # END SOLUTION BLOCK

    # Plotting the PSD
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(time, e_hat)
        plt.title('Residuals')
        plt.ylabel('Temperature [째C]')
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


# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
A = np.column_stack((np.ones(len(data)), np.cos(2*np.pi*time_days/365), np.sin(2*np.pi*time_days/365)))
dom_f = find_frequency(data, time_days, A, fs=fs)
print(f'Dominant Frequency: {dom_f:.2f}')

# check whether there is still a significant frequency in the residuals
find_frequency(data, time_days, np.column_stack((A, np.cos(2*np.pi*time_days), np.sin(2*np.pi*time_days))) , fs=fs, plot=True)
# END SOLUTION BLOCK

# ----------------------------------------
def rewrite_seasonal_comp(ak, bk):
    '''
    Function to rewrite the seasonal component in terms of sin and cos
    ak: seasonal component coefficient for cos
    bk: seasonal component coefficient for sin

    returns: Ak, theta_k
    '''
    # YOUR_CODE_HERE

    # SOLUTION
    Ak = np.sqrt(ak**2 + bk**2)
    theta_k = np.arctan2(-bk, ak)
    return Ak, theta_k
    # END SOLUTION BLOCK

# creating the A matrix of the functional model
# A = YOUR_CODE_HERE
# x_hat, y_hat, e_hat = YOUR_CODE_HERE


# SOLUTION
A = np.column_stack((np.ones(len(data)),
                        np.cos(2*np.pi*1*time_days), np.sin(2*np.pi*1*time_days),
                        np.cos(2*np.pi*time_days/365), np.sin(2*np.pi*time_days/365)))

x_hat, y_hat, e_hat0 = fit_model(data, time_days, A)
# END SOLUTION BLOCK

# Plotting the data and the estimated trend
plt.figure(figsize=(10, 3))
plt.plot(time_days, data, label='Original data')
plt.plot(time_days, y_hat, label='Estimated trend')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [째C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
plt.legend()

# Plotting the residuals
plt.figure(figsize=(10, 3))
plt.plot(time_days, e_hat0)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [째C]')
plt.title('Residuals')
plt.grid(True)

# Extracting the seasonal component coefficients from the estimated parameters
# a_i = YOUR_CODE_HERE
# b_i = YOUR_CODE_HERE
# freqs = YOUR_CODE_HERE


# SOLUTION
a_i = np.array([x_hat[1], x_hat[3]])
b_i = np.array([x_hat[2], x_hat[4]])
freqs = np.array([1, 1/365])
# END SOLUTION BLOCK

print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')

# ----------------------------------------
def A1_matrix(A0, break_point):
    '''
    Function to create the A1 matrix
    A0: A matrix under H0
    break_point: break point location
    return: A1 matrix
    A
    '''
    # create the new column and stack it to the A0 matrix
    # YOUR_CODE_HERE
    
    # SOLUTION
    new_col = np.zeros(A0.shape[0])
    new_col[break_point:] = 1
    A1 = np.column_stack((A0, new_col))
    # END SOLUTION BLOCK
    return A1


def LR(e0, e1, cv=100, verbose=True):
    '''
    Function to perform the LR test
    e0: residuals under H0
    e1: residuals under H1
    cv: critical value
    '''
    # n = YOUR_CODE_HERE
    # SSR0 = YOUR_CODE_HERE
    # SSR1 = YOUR_CODE_HERE
    # test_stat = YOUR_CODE_HERE
    
    # SOLUTION
    n = len(e0)
    SSR0 = e0.T @ e0
    SSR1 = e1.T @ e1
    test_stat = n*np.log(SSR0 / SSR1)
    # END SOLUTION

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
    # initialize the results vector
    # results = YOUR_CODE_HERE
    # find the residuals under H0
    # YOUR_CODE_HERE

    # SOLUTION
    results = np.zeros(len(data))
    _, _, e_hat0 = fit_model(data, time, A)
    # END SOLUTION BLOCK

    # loop over the data points
    for i in range(1, len(data)):
        # create the A1 matrix
        # A1 = YOUR_CODE_HERE

        # SOLUTION
        A1 = A1_matrix(A, i)
        # END SOLUTION BLOCK

        # We need this statement to avoid singular matrices
        if np.linalg.matrix_rank(A1) < A1.shape[1]:
            pass
        else:
            # find the residuals under H1
            # _, _, e_hat1 = YOUR_CODE_HERE
            # test_stat = YOUR_CODE_HERE
            # results[i] = YOUR_CODE_HERE

            # SOLUTION
            _, _, e_hat1 = fit_model(data, time, A1)
            test_stat = LR(e_hat0, e_hat1, verbose=False)
            results[i] = test_stat
            # END SOLUTION BLOCK

    results = np.array(results)
    # finding the offset location. 
    # Offset is the location where the test statistic is maximum

    # location = YOUR_CODE_HERE
    # value = YOUR_CODE_HERE

    # SOLUTION
    location = np.argmax(results)
    value = results[location]
    # END SOLUTION BLOCK

    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(time, results)
        plt.plot(time[location], value, 'ro', label='offset location')
        plt.plot([0, max(time)], [cv, cv], 'k--', label='Critical Value')
        plt.xlabel('Time [days]')
        plt.ylabel('Test Statistic')
        plt.title('LR Test')
        plt.grid(True)
        plt.legend()

    return location, value


# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
A_offset = A.copy()

while True:
    break_point, test_stat = jump_detection(data, time_days, A_offset)
    print(f'Break Point day: {break_point/24} with : {test_stat:.2f}')
    if test_stat < 100:
        break
    A_offset = A1_matrix(A_offset, break_point) 
# END SOLUTION BLOCK

# ----------------------------------------
# A2 = YOUR_CODE_HERE
# x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)

# SOLUTION
A2 = A_offset
x_hat, y_hat, e_hat = fit_model(data, time_days, A2)
# END SOLUTION (PART 1 of 2)

# Plotting the data and the estimated trend
plt.figure(figsize=(10, 3))
plt.plot(time_days, data, label='Original data')
plt.plot(time_days, y_hat, label='Estimated trend')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [째C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
plt.legend()

# Plotting the residuals
plt.figure(figsize=(10, 3))
plt.plot(time_days, e_hat)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [째C]')
plt.title('Residuals')
plt.grid(True)

# Extracting the seasonal component coefficients from the estimated parameters
# a_i = YOUR_CODE_HERE
# b_i = YOUR_CODE_HERE
# freqs = YOUR_CODE_HERE

# SOLUTION
a_i = np.array([x_hat[1], x_hat[3]])
b_i = np.array([x_hat[2], x_hat[4]])
freqs = np.array([1, 1/365])
# END SOLUTION BLOCK

print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')

# ----------------------------------------
# Lets start with the ACF plot
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat, ax=ax, lags=20);
ax.grid()

# ----------------------------------------
def AR1(s, time, plot=True):
    '''
    Function to find the AR(1) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    # y = YOUR_CODE_HERE
    # y_lag_1 = YOUR_CODE_HERE
    # A = np.atleast_2d(y_lag_1).T
    # x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)

    # SOLUTION
    y = s[1:]
    y_lag_1 = s[:-1]
    A = np.atleast_2d(y_lag_1).T
    x_hat, y_hat, e_hat = fit_model(y, time, A)
    # END SOLUTION

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(time[1:], y, label='Original Residuals')
        ax[0].plot(time[1:], y_hat, label='Estimated Residuals')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Temperature [째C]')
        ax[0].set_title('Original Data vs Estimated Data')
        ax[0].grid(True)
        ax[0].legend()
        plot_acf(e_hat, ax=ax[1], lags=20)
        ax[1].grid()
        fig.tight_layout()
        
    print(f'Estimated Parameters:')
    print(f'phi = {x_hat[0]:.4f}')

    return x_hat, e_hat

# Estimating the AR(1) model
# phi_hat_ar1, e_hat_ar1 = AR1(YOUR_CODE_HERE)

# SOLUTION
phi_hat_ar1, e_hat_ar1 = AR1(e_hat, time_days)
# END SOLUTION





# ----------------------------------------
def AR2(s, time, plot=True):
    '''
    Function to find the AR(2) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    # y = YOUR_CODE_HERE
    # y_lag_1 = YOUR_CODE_HERE
    # y_lag_2 = YOUR_CODE_HERE
    # A = YOUR_CODE_HERE
    # x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)

    # SOLUTION
    y = s[2:]
    y_lag_1 = s[1:-1]
    y_lag_2 = s[:-2]
    A = np.column_stack((y_lag_1, y_lag_2))
    x_hat, y_hat, e_hat = fit_model(y, time, A)
    # END SOLUTION

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(time[2:], y, label='Original Residuals')
        ax[0].plot(time[2:], y_hat, label='Estimated Residuals')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Temperature [째C]')
        ax[0].set_title('Original Data vs Estimated Data')
        ax[0].grid(True)
        ax[0].legend()
        plot_acf(e_hat, ax=ax[1], lags=20)
        ax[1].grid()
        fig.tight_layout()

    print(f'Estimated Parameters:')
    print(f'phi_1 = {x_hat[0]:.4f}, phi_2 = {x_hat[1]:.4f}')

    return x_hat, e_hat

# Estimating the AR(2) model
# phi_hat_ar2, e_hat_ar2 = AR2(YOUR_CODE_HERE)

# SOLUTION
phi_hat_ar2, e_hat_ar2 = AR2(e_hat0, time_days)
# END SOLUTION


# ----------------------------------------
# combine ar2 and functional model

# A_final = YOUR_CODE_HERE
# x_hat, y_hat, e_hat_final = fit_model(YOUR_CODE_HERE)

# SOLUTION
A_final = np.column_stack((A2[2:], e_hat[1:-1], e_hat[:-2]))
x_hat, y_hat, e_hat_final = fit_model(data[2:], time_days[2:], A_final, plot=True)
# END SOLUTION

# Plotting the acf of the residuals

# fig, ax = plt.subplots(1, 1, figsize=(10, 3))
# plot_acf(YOUR_CODE_HERE, ax=ax, lags=20);
# ax.grid()

# SOLUTION
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat_final, ax=ax, lags=20);
ax.grid()
# END SOLUTION

# # compute the standard errors
# N = YOUR_CODE_HERE
# p = YOUR_CODE_HERE
# sigma2 = YOUR_CODE_HERE
# Cov = YOUR_CODE_HERE
# se = YOUR_CODE_HERE

# # Extracting the seasonal component coefficients from the estimated parameters
# a_i = YOUR_CODE_HERE
# b_i = YOUR_CODE_HERE
# freqs = YOUR_CODE_HERE

# SOLUTION
# compute the standard errors
N = A_final.shape[0]
p = A_final.shape[1]
sigma2 = np.sum(e_hat_final**2) / (N - p)
Cov = sigma2 * np.linalg.inv(A_final.T @ A_final)
se = np.sqrt(np.diag(Cov))

# Extracting the seasonal component coefficients from the estimated parameters
a_i = np.array([x_hat[1], x_hat[3]])
b_i = np.array([x_hat[2], x_hat[4]])
freqs = np.array([1, 1/365])
# END SOLUTION

# Check if the number of coefficients match the number of frequencies
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


