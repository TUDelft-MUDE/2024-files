import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
from math import ceil, trunc
plt.rcParams.update({'font.size': 14})
T = pd.read_csv('temp.csv', delimiter = ',', parse_dates = True).dropna().reset_index(drop=True)
T.columns=['Date', 'T'] #rename columns
T.head()
T['Date'] = pd.to_datetime(T['Date'], format='mixed')
T['Date']
fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
axes[0].hist(T['T'], label = 'T', density = True, edgecolor = 'darkblue')
axes[0].set_xlabel('PDF')
axes[0].set_xlabel('Maximum daily temperature [degrees]')
axes[0].grid()
axes[0].legend()
axes[0].set_title('(a) Histogram')
axes[1].plot(T['Date'], T['T'],'k', label = 'P')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Maximum daily temperature [degrees]')
axes[1].grid()
axes[1].set_title('(b) Time series')
axes[1].legend()
T['Year'] = T['Date'].dt.year
T['Month'] = T['Date'].dt.month
idx_max = T.groupby(['Year', 'Month'])['T'].idxmax().reset_index(name='Index').iloc[:, 2]
max_list = T.loc[idx_max]
fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
axes[0].hist(T['T'], label = 'T', density = True, edgecolor = 'darkblue')
axes[0].hist(max_list['T'], label = 'Monthly Maxima',  density = True, edgecolor = 'k', alpha = 0.5)
axes[0].set_xlabel('PDF')
axes[0].set_xlabel('Maximum daily temperature [degrees]')
axes[0].grid()
axes[0].set_title('(a) Histograms')
axes[0].legend()
axes[1].plot(T['Date'], T['T'],'k', label = 'T')
axes[1].scatter(max_list['Date'], max_list['T'], 40, 'r', label = 'Monthly Maxima')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Maximum daily temperature [degrees]')
axes[1].set_title('(b) Timeseries')
axes[1].grid()
axes[1].legend()
def ecdf(var):
    x = np.sort(var)
    n = x.size
    y = np.arange(1, n+1) / (n+1)
    return [y, x]
fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
axes[0].hist(T['T'], label = 'T', density = True, edgecolor = 'darkblue')
axes[0].hist(max_list['T'], label = 'Monthly Maxima',  density = True, edgecolor = 'k', alpha = 0.5)
axes[0].set_xlabel('PDF')
axes[0].set_xlabel('Maximum daily temperature [degrees]')
axes[0].grid()
axes[0].legend()
axes[1].step(ecdf(T['T'])[1],
         ecdf(T['T'])[0],'k', label = 'T')
axes[1].step(ecdf(max_list['T'])[1],
         ecdf(max_list['T'])[0],
         'cornflowerblue', label = 'Monthly maxima of T')
axes[1].set_xlabel('Maximum daily temperature [degrees]')
axes[1].set_ylabel('${P[X \leq x]}$')
axes[1].grid()
axes[1].set_yscale('log')
axes[1].legend();
params_T = stats.genextreme.fit(max_list['T'])
print(params_T)
x_range = np.linspace(0.05, 38, 100)
plt.figure(figsize=(10, 6))
plt.step(ecdf(max_list['T'])[1], 1-ecdf(max_list['T'])[0],'cornflowerblue', label = 'Monthly maxima of T')
plt.plot(x_range, 1-stats.genextreme.cdf(x_range, *params_T),
             '--k', label='GEV')
plt.xlabel('Maximum daily temperature [mm]')
plt.ylabel('${P[X > x]}$')
plt.yscale('log') 
plt.grid()
plt.legend();
RT_range = np.linspace(1, 500, 500)
monthly_probs = 1/(RT_range*12)
eval_temp = stats.genextreme.ppf(1-monthly_probs, *params_T)
plt.figure(figsize=(10, 6))
plt.plot(eval_temp, RT_range, 'k')
plt.xlabel('Temperature [mm]')
plt.ylabel('RT [years]')
plt.yscale('log') 
plt.grid()
RT_design = 475
monthly_design = 1/(RT_design*12)
design_T = stats.genextreme.ppf(1-monthly_design, *params_T)
print('The design value of temperature is:',
      f'{design_T:.3f}')
