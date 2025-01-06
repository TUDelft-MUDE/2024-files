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
YOUR_CODE_HERE
N = 100
print(f'{N} floor divided by 2: {N//2}')
print(f'{N+1} floor divided by 2: {(N+1)//2}')
print(f'{N-1} floor divided by 2: {(N-1)//2}')
YOUR_CODE_HERE
plt.figure(figsize=(12,4))
for i, T_meas in enumerate(YOUR_CODE_HERE):
    YOUR_CODE_HERE
    plt.subplot(1, 3, i+1)
    YOUR_CODE_HERE
    plt.grid()
    plt.tight_layout()
plt.legend()
YOUR_CODE_HERE
YOUR_CODE_HERE
YOUR_CODE_HERE
df = pd.read_csv('cantileverbeam_acc50Hz.csv', header=0)
t = np.array(df['time']) #
dat = np.array(df['acceleration']) #
N = len(t)
plt.figure()
plt.plot(t, dat, color='b', label='acceleration signal')
plt.xlabel('time [s]')
plt.ylabel('acceleration [m/s2]')
plt.title('Vertical cantilever beam acceleration')
plt.legend()
T = (t[N-1] - t[0])*N/(N - 1)
dt = T/N
YOUR_CODE_HERE
YOUR_CODE_HERE
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
YOUR_CODE_HERE
