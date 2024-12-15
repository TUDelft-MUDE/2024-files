# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import chi2
from scipy.signal import periodogram

# %%
data = np.loadtxt('atm_data.txt', delimiter=',')
time = data[:, 0]
data = data[:, 1]

dt = time[1] - time[0]
fs = 1 / dt

plt.figure(figsize=(10, 3))
plt.plot(time, data)
plt.xlabel('Time [days]')
plt.ylabel('Atmospheric Pressure [hPa]')
plt.title('2 year of atmospheric pressure data')
plt.grid(True)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def fit_model(data, time, A, plot=False):
    '''
    Function to find the least squares solution of the data
    data: input data
    time: time vector
    A: A-matrix to fit the data
    plot: boolean to plot the results or not
    '''

    
    
    

    
    x_hat = np.linalg.solve(A.T @ A, A.T @ data)
    y_hat = A @ x_hat
    e_hat = data - y_hat
    

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
    
    _, _, e_hat= fit_model(data, time, A)

    N = len(data)

    
    
    
    
    
    
    
    freqs, pxx = periodogram(e_hat, fs=fs, window='boxcar',
                             nfft=N, return_onesided=False,
                             scaling='density')
    

    
    
    
    

    
    
    dominant_frequency, amplitude = freqs[np.argmax(pxx)], np.max(pxx)
    

    
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

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

A = np.column_stack((np.ones(len(data)), time))
dom_f = find_frequency(data, time, A, fs)
print(f'Dominant Frequency: {dom_f*365:.3f} cycle/year')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

A2 = np.column_stack((A, np.cos(2*np.pi*dom_f*time), np.sin(2*np.pi*dom_f*time)))
dom_f2 = find_frequency(data, time, A2, fs)
print(f'Dominant Frequency: {dom_f2:.3f} cycle/day')

# %%

A3 = np.column_stack((A2, np.cos(2*np.pi*dom_f2*time), np.sin(2*np.pi*dom_f2*time)))
dom_f3 = find_frequency(data, time, A3, fs)
print(f'Dominant Frequency: {dom_f3:.3f} cycle/day')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def rewrite_seasonal_comp(ak, bk):
    '''
    Function to rewrite the seasonal component in terms of sin and cos
    ak: seasonal component coefficient for cos
    bk: seasonal component coefficient for sin
    '''
    Ak = np.sqrt(ak**2 + bk**2)
    theta_k = np.arctan2(-bk, ak)
    return Ak, theta_k

A = np.column_stack((   np.ones(len(data)), time,
                        np.cos(2*np.pi*dom_f*time), np.sin(2*np.pi*dom_f*time),
                        np.cos(2*np.pi*dom_f2*time), np.sin(2*np.pi*dom_f2*time)))

x_hat, y_hat, e_hat = fit_model(data, time, A)

a_i = np.array([x_hat[2], x_hat[4]])
b_i = np.array([x_hat[3], x_hat[5]])
freqs = np.array([dom_f, dom_f2])

print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.2f}')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i = i + 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat, ax=ax, lags=20);
ax.set_xlabel('Lags [days]')
ax.grid()

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def AR1(s, time, plot=True):
    '''
    Function to find the AR(1) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    y = s[1:]
    y_lag_1 = s[:-1]
    A = np.atleast_2d(y_lag_1).T
    x_hat, y_hat, e_hat = fit_model(y, time, A)
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(time, y, label='Original Residuals')
        plt.plot(time, y_hat, label='Estimated Residuals')
        plt.xlabel('Time [days]')
        plt.ylabel('Atmospheric Pressure [hPa]')
        plt.title('Original Data vs Estimated Data')
        
        plt.grid(True)
        plt.legend()

    print(f'Estimated Parameters:')
    print(f'phi = {x_hat[0]:.4f}')

    return x_hat, e_hat

# %% [markdown]

# %% [markdown]

# %%
_, e_hat2 = AR1(e_hat, time[1:])

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat2, ax=ax, lags=20);
ax.grid()

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

A_final = np.column_stack((A[1:,:], e_hat[:-1]))
x_hat, y_hat, e_hat_final = fit_model(data[1:], time[1:], A_final, plot=True)

N = A_final.shape[0]
p = A_final.shape[1]
sigma2 = np.sum(e_hat_final**2) / (N - p)
Cov = sigma2 * np.linalg.inv(A_final.T @ A_final)
se = np.sqrt(np.diag(Cov))

a_i = np.array([x_hat[2], x_hat[4]])
b_i = np.array([x_hat[3], x_hat[5]])
freqs = np.array([dom_f, dom_f2])

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

# %% [markdown]

# %% [markdown]

