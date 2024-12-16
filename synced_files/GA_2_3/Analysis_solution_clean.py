
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

T_meas = 5
f_s = 100 #sampling rate [Hz]

t_vec = np.arange(0, T_meas, 1 / f_s) # ends at 4.99, length 500 according to the sample-and-hold convention

A = 1 
f_c = 1 
phi = 5 * np.pi / 180
x = A * np.sin(2 * np.pi * f_c * t_vec + phi)

plt.plot(t_vec, x, color='b', label='signal')
plt.xlabel(r'$t \: [s]$')
plt.ylabel(r'$x(t) \: [V]$')
plt.legend(loc='upper right')
plt.title(fr'Sinusoidal signal with $A$={A} V, $f_c$={f_c} Hz and initial phase $\phi$={phi:.3f} °')
plt.grid()

T_meas = 5
f_s = 100

t_vec = np.arange(0, T_meas, 1/f_s) # ends at 4.99, length 500 according to the sample-and-hold convention

A = 1
f_c = 1
phi = 5 * np.pi / 180
x = A * np.sin(2 * np.pi * f_c * t_vec + phi)

N = len(x)
X_cont = np.fft.fft(x) / N

f_0 = f_s / N
f_vec = np.arange(0, f_s, f_0)

f, axes = plt.subplots(1,2,figsize=(10,5))

axes[0].plot(f_vec, np.abs(X_cont), 'x', color='b', label='Fourier transform')
axes[0].loglog()
axes[0].set_xlabel(r'$f \: \: [Hz]$')
axes[0].set_ylabel(r'$|X(f)| \: [V]$')
axes[0].grid()
axes[0].set_title('Log/Log')

axes[1].plot(f_vec, np.abs(X_cont), 'x', color='b', label='Fourier transform')
axes[1].set_xlabel(r'$f \: \: [Hz]$')
axes[1].set_ylabel(r'$|X(f)| \: [V]$')
axes[1].grid()
axes[1].set_title('Linear')
plt.legend()

print(f_vec[np.abs(X_cont)>0.1])

N = 100
print(f'{N} floor divided by 2: {N//2}')
print(f'{N+1} floor divided by 2: {(N+1)//2}')
print(f'{N-1} floor divided by 2: {(N-1)//2}')

T_lst = [1, 5, 20]
f_s = 100

plt.figure(figsize=(12,4))
for i, T_meas in enumerate(T_lst):
    t_vec = np.arange(0, T_meas, 1/f_s)
    A = 1
    f_c = 1
    phi = 5 * np.pi / 180
    x = A * np.sin(2 * np.pi * f_c * t_vec + phi)
    
    N = len(x)
    X_cont = np.fft.fft(x) / N
    
    f_0 = f_s / N
    f_vec = np.arange(0, f_s, f_0)
    
    X_cont = X_cont[:N//2]
    f_vec = f_vec[:N//2]

    plt.subplot(1, 3, i+1)
    plt.plot(f_vec, np.abs(X_cont), 'x', color='b', label='Fourier transform')
    plt.loglog()
    plt.xlim(0.04, 100)
    plt.ylim(10**(-19), 10)
    plt.xlabel(r'$f \: \: [Hz]$')
    plt.ylabel(r'$|X(f)| \: [V]$')
    plt.grid()
    plt.tight_layout()
plt.legend();

T_meas = 5
f_s = 100

t_vec = np.arange(0, T_meas, 1/f_s) # ends at 4.99, length 500

A = 1
f_c = 1
phi = 5 * np.pi / 180
x = A * np.sin(2 * np.pi * f_c * t_vec + phi)

A_i = 0.1
f_i = 80
x += A_i * np.sin(2 * np.pi * f_i * t_vec)

N = len(x)
X_cont = np.fft.fft(x) / N

f_0 = f_s / N
f_vec = np.arange(0, f_s, f_0)

X_cont = X_cont[:N//2]
f_vec = f_vec[:N//2]

plt.figure(figsize=(12,6))
plt.subplot(211)
plt.plot(t_vec, x, color='b', label='signal')
plt.xlabel(r"$t \: [s]$")
plt.ylabel(r"$x(t) \: [V]$")
plt.grid()
plt.title('Time signal')
plt.legend()

plt.subplot(212)
plt.plot(f_vec, np.abs(X_cont), 'x', color='b', label='Fourier transform')
plt.loglog()
plt.xlim(0.04, 100)
plt.ylim(10**(-19), 10)
plt.xlabel(r"$f \: \: [Hz]$")
plt.ylabel(r"$|X(f)| \: [V]$")
plt.grid()
plt.tight_layout()
plt.title('Amplitude spectrum')
plt.legend();

f_s_lst = [110, 150, 160, 200]

for f_s in f_s_lst:
    T_meas = 5
    # f_s = 100
    
    t_vec = np.arange(0, T_meas, 1/f_s)
    
    A = 1
    f_c = 1
    phi = 5 * np.pi / 180
    x = A * np.sin(2 * np.pi * f_c * t_vec + phi)
    
    A_i = 0.1
    f_i = 80
    x += A_i * np.sin(2 * np.pi * f_i * t_vec)
    
    N = len(x)
    X_cont = np.fft.fft(x) / N
    
    f_0 = f_s / N
    f_vec = np.arange(0, f_s, f_0)
    
    X_cont = X_cont[:N//2]
    f_vec = f_vec[:N//2]
    
    plt.figure(figsize=(12,6))
    plt.suptitle(f'$f_s = {f_s}$ Hz')
    plt.subplot(211)
    plt.plot(t_vec, x, color='b', label='signal')
    plt.xlabel(r'$t \: [s]$')
    plt.ylabel(r'$x(t) \: [V] $')
    plt.grid()
    plt.legend()
    plt.subplot(212)
    plt.plot(f_vec, np.abs(X_cont), 'x', color='b', label='Fourier transform')
    plt.loglog()
    plt.xlim(0.04, 100)
    plt.ylim(10**(- 19), 10)
    plt.xlabel(r'$f \: \: [Hz]$')
    plt.ylabel(r'$|X(f)| \: [V]$')
    plt.grid()
    plt.tight_layout()
    plt.legend()

T_meas = 50
f_s = 100

t_vec = np.arange(0, T_meas, 1/f_s)

x_0 = 1
zeta = 0.05
omega_0 = 10 * np.pi
omega_d = omega_0 * np.sqrt(1 - zeta**2)
x = x_0 / (np.sqrt(1 - zeta**2)) * np.exp(-zeta * omega_0 * t_vec) * np.sin(omega_d * t_vec)

N = len(x)
X_cont = np.fft.fft(x) / N

f_0 = f_s / N
f_vec = np.arange(0, f_s, f_0)

X_cont = X_cont[:N//2]
f_vec = f_vec[:N//2]

plt.figure(figsize=(12,6))
plt.subplot(211)
plt.plot(t_vec, x, color='b', label='signal')
plt.xlabel(r"$t \: [s]$")
plt.ylabel(r"$x(t) \: [V] $")
plt.title('Time signal')
plt.grid()
plt.legend()

plt.subplot(212)
plt.plot(f_vec, np.abs(X_cont), 'x', color='b', label='Fourier transfrom')
plt.loglog()
plt.xlim(0.04, 100)
plt.ylim(10**(-7), 10)
plt.xlabel(r"$f \: \: [Hz]$")
plt.ylabel(r"$|X(f)| \: [V]$")
plt.title('Amplitude spectrum')
plt.grid()
plt.tight_layout()
plt.legend()

print(f_vec[np.argmax(np.abs(X_cont))])

df = pd.read_csv('cantileverbeam_acc50Hz.csv', header=0)

t = np.array(df['time']) #
dat = np.array(df['acceleration']) #

N = len(t)

T = (t[N-1] - t[0]) * N / (N - 1)
dt = T / N
plt.figure()
plt.plot(t, dat, color='b', label='acceleration signal')
plt.xlabel('time [s]')
plt.ylabel('acceleration [m/s2]')
plt.title('Vertical cantilever beam acceleration')
plt.legend()

A = np.column_stack((np.ones(N), t - t[0]))

xhat = np.linalg.inv((A.T)@A)@A.T@dat
yhat = A@xhat
ehat = dat - yhat

plt.figure()
plt.plot(t, ehat, color='b', label='detrended signal')
plt.xlabel('time [s]')
plt.ylabel('detrended accelerations [m/s2]')
plt.title('Detrended vertical cantilever beam acceleration')
plt.legend();

Fs = 1 / dt
f0 = 1 / T
print(Fs)
print(f0)

f = np.concatenate((np.arange(-Fs / 2 + f0 / 2, 0, f0), np.arange(0, Fs / 2, f0)))
print(f)

Z = np.fft.fft(ehat) * dt
psd = (np.abs(Z))**2/T 
plt.figure()
plt.plot(f,np.fft.fftshift(psd), color='b', label='psd')
print(Z)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [m2/s4/Hz]')
plt.title('Power Spectral Density (PSD)')
plt.legend();

data = pd.read_csv('CSIRO_Alt_seas_inc.txt', names=['month', 'sl'])
data.head()

t = data.iloc[:, 0] - data.iloc[0, 0]

N = len(t)

T = (t[N - 1] - t[0]) * N / (N - 1)

dt = T / N

y = data.iloc[:,1]

plt.plot(data.iloc[:,0],y, color='b', label='sea level')
plt.xlabel('time [yr]')
plt.ylabel('sea-level height [mm]')
plt.title('Global Mean Sea-Level (GMSL) rise')
plt.legend();

A = np.ones((N, 2))
A[:,1] = t

xhat = (np.linalg.inv(A.T @ A) @ A.T) @ y

yhat = A @ xhat

ehat = y - yhat

plt.plot(data.iloc[:,0], ehat, color='b', label='detrended signal')
plt.xlabel('time [yr]')
plt.ylabel('detrended sea-level height [mm]')
plt.title('Detrened Global Mean Sea-Level')
plt.legend()

Fs = 1 / dt

NFFT = N

X = dt * np.fft.fft(ehat, NFFT)

f0 = 1 / (NFFT * dt)

f = np.concatenate((np.arange(-Fs / 2 + f0 / 2, 0, f0), np.arange(0, Fs / 2 , f0))) #- f0 / 4 

plt.plot(f, np.fft.fftshift((abs(X))**2 / T), color='b', label='psd')
plt.xlabel(r'frequency [$yr^{-1}$]')
plt.ylabel(r'PSD [$mm^2$ yr]')
plt.title('Power Spectral Density of GMSL data')
plt.legend();

